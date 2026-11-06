import streamlit as st
from stats_dict import stats_dict

# Sample data
query_data = {i: line.strip() for i, line in enumerate(open("test.jsonl"))}

# Initialize selected keys
selected_keys = {}


# Function to get lines
def get_lines(stats_key, keys):
    indices = []
    for key in keys:
        indices.extend(stats_dict[stats_key]["_reverse_lookup"][key])
    return "\n".join([query_data[i] for i in indices])


# Function to render dropdown and button
def render_dropdown_and_button(stats_key):
    st.subheader(f"Stats for `{stats_key}`")
    st.json(stats_dict[stats_key]["counter"])
    st.json(
        {k: v for k, v in stats_dict[stats_key].items() if isinstance(v, (int, float))}
    )
    st.subheader("Histogram")
    st.bar_chart(stats_dict[stats_key]["counter"], use_container_width=True)

    options = list(stats_dict[stats_key]["counter"].keys())
    selected_keys[stats_key] = st.multiselect(
        f"View samples with {stats_key}",
        options,
        default=selected_keys.get(stats_key, []),
    )
    st.code(get_lines(stats_key, selected_keys[stats_key]))


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a page:",
    ["Validation Stats", "Individual Path Views"],
)

# Main Streamlit App
st.title("Structured Output Evaluation")

# Validation Stats
if page == "Validation Stats":
    st.header("Validation Stats")
    for key in [k for k in stats_dict.keys() if k.startswith("_")]:
        render_dropdown_and_button(key)

# Individual Path Views
elif page == "Individual Path Views":
    st.header("Individual Path Views")
    path = st.selectbox(
        "Choose a path:",
        [key for key in stats_dict.keys() if not key.startswith("_")],
    )
    if "counter" in stats_dict[path]:
        render_dropdown_and_button(path)
