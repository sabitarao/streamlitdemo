import streamlit as st
import json
import os
import re
import graphviz

# Page Config
st.set_page_config(
    page_title="AI Ontology Retrieval Demo",
    page_icon=None,
    layout="wide"
)

# --- Helper Functions ---
def clean_mdx(text):
    """Removes MDX tags and formats them for Markdown display."""
    if not text: return ""
    
    # Replace <Card title="..."> with **Title**
    text = re.sub(r'<Card\s+title="([^"]+)"[^>]*>', r'**\1**\n', text)
    
    # Replace <Accordion title="..."> with ### Title
    text = re.sub(r'<Accordion\s+title="([^"]+)"[^>]*>', r'### \1\n', text)
    
    # Replace <Warning> with blockquote
    text = re.sub(r'<Warning>', r'> **Warning**\n', text)
    
    # Replace <Tip> with blockquote
    text = re.sub(r'<Tip>', r'> **Tip**\n', text)
    
    # Remove other opening tags (CardGroup, CodeGroup, etc)
    text = re.sub(r'<[A-Za-z]+[^>]*>', '', text)
    
    # Remove closing tags
    text = re.sub(r'</[A-Za-z]+>', '', text)
    
    return text

# --- Load Data ---
@st.cache_data
def load_data():
    # Load Entities
    entities_path = os.path.join("audits", "consolidated-ontology.json")
    relationships_path = os.path.join("audits", "relationships.json")
    
    entities = []
    metadata = {}
    relationships = []
    
    try:
        with open(entities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entities = data.get('entities', [])
        metadata = data.get('metadata', {})
        
        # Post-process entities for display
        for item in entities:
            # Map new schema class to the 'category' field used by the UI
            item['category'] = item.get('entity_class', 'Uncategorized')
            
            # --- ENHANCEMENT: Smart Title & Content ---
            # 1. Determine Display Title
            if 'title' in item:
                item['display_title'] = item['title']
            elif 'term' in item:
                item['display_title'] = item['term']
            elif 'platform' in item:
                item['display_title'] = item['platform']
            elif 'activity' in item:
                item['display_title'] = item['activity']
            else:
                # Fallback to ID, capitalized and spaces replaced
                item['display_title'] = item.get('id', 'Untitled').replace('-', ' ').title()
            
            # 2. Determine Display Content
            content_parts = []
            
            # Add definition/purpose if available (crucial for metadata-only items)
            if 'definition' in item:
                content_parts.append(f"**Definition:** {item['definition']}")
            if 'purpose' in item:
                content_parts.append(f"**Purpose:** {item['purpose']}")
            if 'description' in item:
                content_parts.append(f"**Description:** {item['description']}")
            if 'prevents' in item:
                content_parts.append(f"**Prevents:** {item['prevents']}")
            
            # Add the main content if it exists
            if 'content' in item:
                content_parts.append(clean_mdx(item['content']))
            
            if content_parts:
                item['display_content'] = "\n\n".join(content_parts)
            else:
                item['display_content'] = "*No detailed content available for this item.*"

    except FileNotFoundError:
        st.error(f"Could not find {entities_path}")
        
    try:
        with open(relationships_path, 'r', encoding='utf-8') as f:
            rel_data = json.load(f)
            raw_relationships = rel_data.get('relationships', [])
            
            # Robustness Fix: Handle cases where target might be a list (legacy data issue)
            for rel in raw_relationships:
                target = rel.get('target')
                source = rel.get('source')
                
                # Skip invalid entries
                if not source or not target:
                    continue
                    
                if isinstance(target, list):
                    # Flatten list targets into multiple relationships
                    for t in target:
                        if isinstance(t, str):
                            new_rel = rel.copy()
                            new_rel['target'] = t
                            relationships.append(new_rel)
                elif isinstance(target, str):
                    relationships.append(rel)
                    
    except FileNotFoundError:
        st.warning(f"Could not find {relationships_path}. Graph features will be limited.")

    return entities, metadata, relationships

entities, metadata, relationships = load_data()

# --- Search Logic ---
def search_entities(query, entities, selected_types, selected_modules):
    results = []
    
    # 1. Preprocess Query (Tokenization & Stop Words)
    stop_words = {'what', 'does', 'mean', 'how', 'is', 'the', 'a', 'an', 'and', 'or', 'for', 'of', 'in', 'to', 'do', 'can', 'i'}
    raw_tokens = set(re.findall(r'\w+', query.lower()))
    keywords = raw_tokens - stop_words
    
    # If all words were stop words (e.g. "what is it"), keep original to avoid empty search
    if not keywords:
        keywords = raw_tokens

    for entity in entities:
        # 2. Filter by Type/Module
        if selected_types and entity['category'] not in selected_types:
            continue
        if selected_modules and entity['source_module'] not in selected_modules:
            continue
            
        score = 0
        match_reason = []
        
        # 3. Scoring Logic (Hybrid: Exact Phrase + Keyword Match)
        
        # Helper to check text
        def check_text(text, weight, reason_label):
            local_score = 0
            if not text: return 0
            text_lower = text.lower()
            
            # A. Exact Phrase Bonus
            if query.lower() in text_lower:
                local_score += weight * 2
                if reason_label not in match_reason:
                    match_reason.append(f"Exact match in {reason_label}")
            
            # B. Keyword Overlap
            matches = 0
            for word in keywords:
                # Use regex word boundary to avoid substring matches (e.g. "ai" in "main")
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    matches += 1
            
            if matches > 0:
                local_score += matches * weight
                # Only add reason if we haven't added the exact match one
                if matches == len(keywords) and f"Exact match in {reason_label}" not in match_reason:
                     match_reason.append(f"All keywords in {reason_label}")
                elif f"Exact match in {reason_label}" not in match_reason and f"Keywords in {reason_label}" not in match_reason:
                     match_reason.append(f"Keywords in {reason_label}")
            
            return local_score

        # A. Retrieval Questions (Highest Value - Intent)
        if 'retrievalQuestions' in entity:
            for q in entity['retrievalQuestions']:
                score += check_text(q, 5, "Intent") # High weight for questions
        
        # B. Title Match (High Value)
        score += check_text(entity.get('display_title', ''), 4, "Title")
            
        # C. Definition Match (Medium Value)
        score += check_text(entity.get('definition', ''), 3, "Definition")
            
        # D. Content Match (Low Value)
        score += check_text(entity.get('display_content', ''), 1, "Content")

        if score > 0:
            # Normalize reasons to avoid duplicates
            unique_reasons = list(set(match_reason))
            results.append({
                "entity": entity,
                "score": score,
                "reasons": unique_reasons
            })
            
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    if st.button("Reset Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    # API Key Input
    api_key = st.text_input("OpenAI API Key (Optional)", type="password", help="Enter your key to enable real AI generation. Otherwise, the app runs in 'Inspection Mode'.")
    
    st.divider()
    
    st.header("Ontology Filters")
    st.info("Filter the knowledge base to narrow down retrieval.")
    
    # Extract unique values for filters
    all_types = sorted(list(set(e['category'] for e in entities)))
    all_modules = sorted(list(set(e['source_module'] for e in entities)))
    
    selected_types = st.multiselect("Entity Type", all_types, default=None)
    selected_modules = st.multiselect("Source Module", all_modules, default=None)
    
    st.divider()
    st.caption(f"Index Version: {metadata.get('version', 'N/A')}")
    st.caption(f"Total Entities: {len(entities)}")
    st.caption(f"Total Relationships: {len(relationships)}")

# Main Content
st.title("AI-for-IA RAG System")
st.markdown("""
This is a **Retrieval Augmented Generation (RAG)** system powered by a structured Ontology.
It retrieves exact content chunks from the `ai-for-ia` course modules to answer your questions.
""")

# Create Tabs
tab1, tab2 = st.tabs(["Chat & Retrieval", "Knowledge Graph"])

with tab1:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Top Search Interface ---
    
    # 1. Prepare Autocomplete Options
    # Group by category for better organization
    options_map = {
        "Framework": [], "Principle": [], "Concept": [], "Tool": [], 
        "Activity": [], "Artifact": [], "Metric": [], "Uncategorized": []
    }
    questions = []

    for e in entities:
        cat = e.get('category', 'Uncategorized')
        title = e.get('display_title', 'Untitled')
        
        if cat in options_map:
            options_map[cat].append(title)
        
        # Add Questions
        if 'retrievalQuestions' in e:
            for q in e['retrievalQuestions']:
                questions.append(q)

    # Flatten into a sorted list with priority
    final_options = []
    final_options.extend(sorted(list(set(questions)))) # Questions first
    
    # Priority Order: Frameworks > Principles > Activities > Concepts > Tools > Artifacts > Metrics
    final_options.extend(sorted(list(set(options_map["Framework"]))))
    final_options.extend(sorted(list(set(options_map["Principle"]))))
    final_options.extend(sorted(list(set(options_map["Activity"]))))
    final_options.extend(sorted(list(set(options_map["Concept"]))))
    final_options.extend(sorted(list(set(options_map["Tool"]))))
    final_options.extend(sorted(list(set(options_map["Artifact"]))))
    final_options.extend(sorted(list(set(options_map["Metric"]))))
    
    autocomplete_options = final_options

    # 2. Search Logic Callbacks
    def update_from_select():
        if st.session_state.search_selectbox:
            st.session_state.triggered_query = st.session_state.search_selectbox
            st.session_state.search_input = "" # Clear text input

    def update_from_text():
        if st.session_state.search_input:
            st.session_state.triggered_query = st.session_state.search_input
            st.session_state.search_selectbox = None # Clear selectbox

    # 3. Search UI
    st.markdown("### Search Knowledge Base")
    col_search_1, col_search_2 = st.columns([1, 1])
    
    with col_search_1:
        st.selectbox(
            "Predictive Search",
            options=autocomplete_options,
            index=None,
            placeholder="Select a topic or question.",
            key="search_selectbox",
            on_change=update_from_select,
            label_visibility="collapsed"
        )
        

    with col_search_2:
        st.text_input(
            "Custom Search",
            key="search_input",
            placeholder="Or type a custom question...",
            on_change=update_from_text,
            label_visibility="collapsed"
        )
        

    # Suggestions (Always available, collapsible)
    with st.expander("Suggestions for Beginners", expanded=(len(st.session_state.messages)==0)):
        suggestion_cols = st.columns(2)
        suggestions = [
            "What is AI for IA?",
            "What is the RICE framework?",
            "How do I define a role in a prompt?",
            "What are the capabilities of LLMs for IA?"
        ]
        
        for i, q in enumerate(suggestions):
            if suggestion_cols[i % 2].button(q, use_container_width=True, key=f"suggestion_{i}"):
                st.session_state.triggered_query = q
                st.rerun()

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New Search ---
    if "triggered_query" in st.session_state and st.session_state.triggered_query:
        prompt = st.session_state.triggered_query
        del st.session_state.triggered_query
        
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Retrieval (The "R" in RAG)
        with st.chat_message("assistant"):
            # Create a status container to show the "Thinking" process
            with st.status("Retrieving & Analyzing...", expanded=True) as status:
                
                # A. Search the Index
                st.write("Searching knowledge base...")
                results = search_entities(prompt, entities, selected_types, selected_modules)
                top_results = results[:3] # Get top 3 most relevant chunks
                
                if not top_results:
                    status.update(label="No results found", state="error")
                    st.error("I couldn't find any relevant information in the index.")
                    st.stop()
                    
                # B. Show Retrieval Evidence
                st.write(f"Found {len(results)} potential matches. Using top {len(top_results)}:")
                for res in top_results:
                    st.markdown(f"- **{res['entity'].get('display_title', 'Untitled')}** (Relevance Score: {res['score']})")
                    st.caption(f"  *Reason: {', '.join(res['reasons'])}*")
                
                # C. Assemble Context
                context_text = ""
                for res in top_results:
                    context_text += f"\n---\nSOURCE: {res['entity'].get('display_title', 'Untitled')} ({res['entity'].get('source_module', 'Unknown')})\nCONTENT:\n{res['entity'].get('display_content', '')}\n"
                
                status.update(label="Context Retrieved", state="complete")

            # 3. Generation (The "G" in RAG)
            
            if api_key:
                # REAL MODE: Call OpenAI
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    
                    system_prompt = """You are an expert Information Architect assistant. 
                    Answer the user's question using ONLY the provided context. 
                    If the answer is not in the context, say you don't know.
                    Cite your sources (e.g., 'According to the LLM Capability Model...')."""
                    
                    stream = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"}
                        ],
                        stream=True,
                    )
                    
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")
                    st.info("Falling back to Inspection Mode.")
            
            if not api_key:
                # INSPECTION MODE: Show the Retrieved Content directly
                st.success("**Retrieval Complete** (No API Key - Inspection Mode)")
                
                # Optional: Show the prompt for technical inspection
                with st.expander("View Technical Prompt (For Developers)", expanded=True):
                    st.markdown("This is the exact prompt that would be sent to the LLM if an API key were present:")
                    final_prompt = f"""
**System:** You are an expert Information Architect. Answer using ONLY the context below.

**User:** {prompt}

**Context (Retrieved from Ontology):**
{context_text}
                    """
                    st.code(final_prompt, language="text")
                
                st.session_state.messages.append({"role": "assistant", "content": "I have displayed the relevant content chunks above. To have me synthesize a direct answer, please provide an OpenAI API Key."})

with tab2:
    st.header("Ontology Knowledge Graph")
    st.info("Visualize the relationships between concepts, frameworks, and principles.")
    
    # Legend
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
        <span style="background-color: lightcoral; color: #000; padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em;">Framework</span>
        <span style="background-color: lightgreen; color: #000; padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em;">Principle</span>
        <span style="background-color: lightyellow; color: #000; padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em;">Concept</span>
        <span style="background-color: lavender; color: #000; padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em;">Activity</span>
        <span style="background-color: gold; color: #000; padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em;">Selected / Result</span>
    </div>
    """, unsafe_allow_html=True)

    # Check for active search query
    last_query = None
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_query = msg["content"]
                break

    # Graph Controls
    col1, col2 = st.columns(2)
    with col1:
        # Add "Search Results" option if a query exists
        options = ["Module View", "Entity Neighborhood"]
        if last_query:
            options.insert(0, "Search Results (Current)")
        
        graph_mode = st.selectbox("Visualization Mode", options)
    
    # Helper to get node color
    def get_node_color(entity_id, highlight=False):
        if highlight: return "gold"
        
        entity = next((e for e in entities if e['id'] == entity_id), None)
        if not entity: return "white"
        
        cat = entity.get('category', '')
        if cat == 'Framework': return "lightcoral"
        elif cat == 'Principle': return "lightgreen"
        elif cat == 'Concept': return "lightyellow"
        elif cat == 'Activity': return "lavender"
        return "lightblue"

    # Helper to get node label
    def get_node_label(entity_id):
        entity = next((e for e in entities if e['id'] == entity_id), None)
        return entity.get('display_title', entity_id) if entity else entity_id

    if graph_mode == "Search Results (Current)" and last_query:
        with col2:
            st.markdown(f"**Showing context for:** *\"{last_query}\"*")

        # Run a quick search to get the relevant entities
        search_results = search_entities(last_query, entities, None, None)
        top_nodes = search_results[:5] # Top 5 most relevant
        
        if top_nodes:
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')
            
            top_node_ids = [res['entity']['id'] for res in top_nodes]
            nodes_to_include = set(top_node_ids)
            edges_to_draw = []
            
            # Find connections in relationships list
            for rel in relationships:
                src = rel['source']
                dst = rel['target']
                
                # If source is in top nodes, add target and edge
                if src in top_node_ids:
                    nodes_to_include.add(dst)
                    edges_to_draw.append(rel)
                
                # If target is in top nodes, add source and edge
                elif dst in top_node_ids:
                    nodes_to_include.add(src)
                    edges_to_draw.append(rel)

            # Draw Nodes
            for node_id in nodes_to_include:
                label = get_node_label(node_id)
                color = get_node_color(node_id, highlight=(node_id in top_node_ids))
                graph.node(node_id, label=label, style='filled', fillcolor=color, shape='box')
            
            # Draw Edges
            for rel in edges_to_draw:
                graph.edge(rel['source'], rel['target'], label=rel['type'], fontsize='10')
            
            st.graphviz_chart(graph)
        else:
            st.warning("No entities found for this query.")

    elif graph_mode == "Module View":
        with col2:
            selected_graph_module = st.selectbox("Select Module", all_modules)
        
        if selected_graph_module:
            # Filter entities for this module
            module_entities = [e for e in entities if e['source_module'] == selected_graph_module]
            module_entity_ids = set(e['id'] for e in module_entities)
            
            # Create Graph
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')
            
            # Add Nodes
            for entity in module_entities:
                label = entity.get('display_title', entity['id'])
                color = get_node_color(entity['id'])
                graph.node(entity['id'], label=label, style='filled', fillcolor=color, shape='box')
            
            # Add Edges (Internal to module)
            for rel in relationships:
                if rel['source'] in module_entity_ids and rel['target'] in module_entity_ids:
                    graph.edge(rel['source'], rel['target'], label=rel['type'], fontsize='10')
            
            st.graphviz_chart(graph)
            
    elif graph_mode == "Entity Neighborhood":
        with col2:
            # Search for an entity
            # Create a list of titles for the selectbox
            entity_options = {f"{e.get('display_title', 'Untitled')} ({e['id']})": e['id'] for e in entities}
            selected_option = st.selectbox("Select Entity", options=list(entity_options.keys()))
            
        if selected_option:
            target_id = entity_options[selected_option]
            target_entity = next((e for e in entities if e['id'] == target_id), None)
            
            if target_entity:
                st.markdown(f"### Neighborhood: {target_entity.get('display_title')}")
                
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                
                nodes_to_include = {target_id}
                edges_to_draw = []
                
                # Find all connected edges
                for rel in relationships:
                    if rel['source'] == target_id:
                        nodes_to_include.add(rel['target'])
                        edges_to_draw.append(rel)
                    elif rel['target'] == target_id:
                        nodes_to_include.add(rel['source'])
                        edges_to_draw.append(rel)
                
                # Draw Nodes
                for node_id in nodes_to_include:
                    label = get_node_label(node_id)
                    color = get_node_color(node_id, highlight=(node_id == target_id))
                    graph.node(node_id, label=label, style='filled', fillcolor=color, shape='box')
                
                # Draw Edges
                for rel in edges_to_draw:
                    graph.edge(rel['source'], rel['target'], label=rel['type'], fontsize='10')
                    
                st.graphviz_chart(graph)
