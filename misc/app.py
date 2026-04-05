import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

print("Loading finished.")

print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

STYLE = """
.custom-container {
	display: grid;
	align-items: center;
    margin: 0!important;
    overflow-y: hidden;
}
.prose ul ul {
    font-size: 10px!important;
}
.prose li {
    margin-bottom: 0!important;
}
.prose table {
    margin-bottom: 0!important;
}
.prose td, th {
    padding-left: 2px;
    padding-right: 2px;
    padding-top: 0;
    padding-bottom: 0;
    text-wrap:nowrap;
}
.tree {
	padding: 0px;
	margin: 0!important;
	box-sizing: border-box;
    font-size: 10px;
	width: 100%;
	height: auto;
	text-align: center;
    display:inline-block;
    padding-bottom: 10px!important;
}
#root {
    display: inline-grid!important;
    width:auto!important;
    min-width: 220px;
}
.tree ul {
    padding-left: 20px;
    position: relative;
    transition: all 0.5s ease 0s;
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 0px !important;
}
.tree li {
    display: flex;
    text-align: center;
    list-style-type: none;
    position: relative;
    padding-left: 20px;
    transition: all 0.5s ease 0s;
    flex-direction: row;
    justify-content: start;
    align-items: center;
}
.tree li::before, .tree li::after {
    content: "";
    position: absolute;
    left: 0px;
    border-left: 1px solid var(--body-text-color);
    width: 20px;
}
.tree li::before {
    top: 0;
    height:50%;
}
.tree li::after {
    top: 50%;
    height: 55%;
    bottom: auto;
    border-top: 1px solid var(--body-text-color);
}
.tree li:only-child::after, li:only-child::before {
    display: none;
}
.tree li:first-child::before, .tree li:last-child::after {
    border: 0 none;
}
.tree li:last-child::before {
	border-bottom: 1px solid var(--body-text-color);
	border-radius: 0px 0px 0px 5px;
	-webkit-border-radius: 0px 0px 0px 5px;
	-moz-border-radius: 0px 0px 0px 5px;
}
.tree li:first-child::after {
	border-radius: 5px 0 0 0;
	-webkit-border-radius: 5px 0 0 0;
	-moz-border-radius: 5px 0 0 0;
}
.tree ul ul::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    border-top: 1px solid var(--body-text-color);
    width: 20px;
    height: 0;
}
.tree ul:has(> li:only-child)::before {
    width:40px;
}
.child:before {
    border-right: 2px solid var(--body-text-color);
    border-bottom: 2px solid var(--body-text-color);
    content: "";
    position: absolute;
    width: 10px;
    left: 8px;
    height: 10px;
    top: 50%;
    margin-top: -5px;
    transform: rotate(315deg);
}
.tree li a {
	border: 1px solid var(--body-text-color);
	padding: 5px;
	border-radius: 5px;
	text-decoration-line: none;
	border-radius: 5px;
	transition: .5s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    overflow: hidden;
}
.tree li a span {
	padding: 5px;
	font-size: 12px;
	letter-spacing: 1px;
	font-weight: 500;
}
/*Hover-Section*/
.tree li a:hover, .tree li a:hover+ul li a {
	background: var(--primary-500);
}
.tree li a:hover+ul li::after, .tree li a:hover+ul li::before, .tree li a:hover+ul::before, .tree li a:hover+ul ul::before, .tree li a:hover+ul a::before {
	border-color: var(--primary-500);
}
.chosen-token {
    background-color: var(--primary-400);
}
.chosen-token td, .chosen-token tr {
    color: black!important;
}
.end-of-text {
    width:auto!important;
}
.nonfinal {
    width:280px;
    min-width: 280px;
}
.selected-sequence {
    background-color: var(--secondary-500);
}
.nonselected-sequence {
    background-color: var(--primary-500);
}
.nopadding {
    padding-left: 0;
}
"""


def clean(s):
    return s.replace("\n", r"\n").replace("\t", r"\t").strip()


def generate_markdown_table(
    scores, previous_cumul_score, score_divider, top_k=4, chosen_tokens=None
):
    markdown_table = """
    <table>
        <tr>
            <th><b>Token</b></th>
            <th><b>Step score</b></th>
            <th><b>Total score</b></th>
        </tr>"""
    for token_idx in np.array(np.argsort(scores)[-top_k:])[::-1]:
        token = tokenizer.decode([token_idx])
        item_class = ""
        if chosen_tokens and token in chosen_tokens:
            item_class = "chosen-token"
        markdown_table += f"""
        <tr class={item_class}>
            <td>{clean(token)}</td>
            <td>{scores[token_idx]:.4f}</td>
            <td>{(scores[token_idx] + previous_cumul_score)/score_divider:.4f}</td>
        </tr>"""
    markdown_table += """
    </table>"""
    return markdown_table


def generate_nodes(node, step):
    """Recursively generate HTML for the tree nodes."""
    token = tokenizer.decode([node.current_token_ix])

    if node.is_final:
        if node.is_selected_sequence:
            selected_class = "selected-sequence"
        else:
            selected_class = "nonselected-sequence"
        return f"<li> <a class='end-of-text child {selected_class}'> <span> <b>{clean(token)}</b> <br>Total score: {node.total_score:.2f}</span> </a> </li>"

    html_content = (
        f"<li> <a class='nonfinal child'> <span> <b>{clean(token)}</b> </span>"
    )
    if node.table is not None:
        html_content += node.table
    html_content += "</a>"

    if len(node.children.keys()) > 0:
        html_content += "<ul> "
        for token_ix, subnode in node.children.items():
            html_content += generate_nodes(subnode, step=step + 1)
        html_content += "</ul>"
    html_content += "</li>"

    return html_content


def generate_html(start_sentence, original_tree):
    html_output = f"""<div class="custom-container">
				<div class="tree">
                <ul> <li> <a id='root' class="nopadding"> <span> <b>{start_sentence}</b> </span> {original_tree.table} </a>"""
    html_output += "<ul> "
    for subnode in original_tree.children.values():
        html_output += generate_nodes(subnode, step=1)
    html_output += "</ul>"
    html_output += """
        </li> </ul>
        </div>
    </body>
    """
    return html_output


import pandas as pd
from typing import Dict
from dataclasses import dataclass


@dataclass
class BeamNode:
    current_token_ix: int
    cumulative_score: float
    children_score_divider: float
    table: str
    current_sequence: str
    children: Dict[int, "BeamNode"]
    total_score: float
    is_final: bool
    is_selected_sequence: bool


def generate_beams(n_beams, start_sentence, scores, length_penalty, decoded_sequences):
    original_tree = BeamNode(
        cumulative_score=0,
        current_token_ix=None,
        table=None,
        current_sequence=start_sentence,
        children={},
        children_score_divider=(1 ** length_penalty),
        total_score=None,
        is_final=False,
        is_selected_sequence=False,
    )
    beam_trees = [original_tree] * n_beams
    generation_length = len(scores)

    for step, step_scores in enumerate(scores):

        # Gather all possible descendants for each beam
        (
            top_token_indexes,
            top_cumulative_scores,
            beam_indexes,
            current_sequence,
            top_tokens,
            token_scores,
        ) = ([], [], [], [], [], [])

        score_idx = 0
        for beam_ix in range(len(beam_trees)):
            current_beam = beam_trees[beam_ix]

            # skip if the beam is already final
            if current_beam.is_final:
                continue
                
            # Get top cumulative scores for the current beam
            current_top_token_indexes = list(
                np.array(scores[step][score_idx].argsort()[-n_beams:])[::-1]
            )
            top_token_indexes += current_top_token_indexes
            token_scores += list(np.array(scores[step][score_idx][current_top_token_indexes]))
            top_cumulative_scores += list(
                np.array(scores[step][score_idx][current_top_token_indexes])
                + current_beam.cumulative_score
            )
            beam_indexes += [beam_ix] * n_beams
            current_sequence += [beam_trees[beam_ix].current_sequence] * n_beams
            top_tokens += [tokenizer.decode([el]) for el in current_top_token_indexes]
            score_idx += 1

        top_df = pd.DataFrame.from_dict(
            {
                "token_index": top_token_indexes,
                "cumulative_score": top_cumulative_scores,
                "beam_index": beam_indexes,
                "current_sequence": current_sequence,
                "token": top_tokens,
                "token_score": token_scores,
            }
        )
        maxes = top_df.groupby(["token_index", "current_sequence"])[
            "cumulative_score"
        ].idxmax()

        top_df = top_df.loc[maxes]

        # Sort all top probabilities and keep top n_beams * 2 (* 2 because each beam may end this iteration, and we
        # want to keep at least `n_beams` beams alive)
        top_df_selected = top_df.sort_values("cumulative_score", ascending=False).iloc[
            :n_beams * 2
        ]
        beams_to_keep = 0
        unfinished_beams = 0
        for _, row in top_df_selected.iterrows():
            beams_to_keep += 1
            current_token_choice_ix = row["token_index"]
            is_final = step == len(scores) - 1 or current_token_choice_ix == tokenizer.eos_token_id
            if not is_final:
                unfinished_beams += 1
            if unfinished_beams >= n_beams:
                break
            if step == generation_length - 1 and beams_to_keep == n_beams:
                break
        top_df_selected_filtered = top_df_selected.iloc[:beams_to_keep]

        # Write the scores table in each beam tree
        score_idx = 0
        for beam_ix in range(len(beam_trees)):
            current_beam = beam_trees[beam_ix]
            if current_beam.table is None:
                selected_tokens = top_df_selected_filtered.loc[
                    top_df_selected_filtered["current_sequence"] == current_beam.current_sequence
                ]
                markdown_table = generate_markdown_table(
                    step_scores[score_idx, :],
                    current_beam.cumulative_score,
                    current_beam.children_score_divider,
                    chosen_tokens=list(selected_tokens["token"].values),
                )
                beam_trees[beam_ix].table = markdown_table
            if not current_beam.is_final:
                score_idx = min(score_idx + 1, n_beams - 1)

        # Add new children to each beam
        cumulative_scores = [beam.cumulative_score for beam in beam_trees]
        for _, row in top_df_selected_filtered.iterrows():
            # Update the source tree
            source_beam_ix = int(row["beam_index"])
            current_token_choice_ix = row["token_index"]
            current_token_choice = tokenizer.decode([current_token_choice_ix])
            token_scores = row["token_score"]

            cumulative_score = cumulative_scores[source_beam_ix] + np.asarray(token_scores)
            current_sequence = (
                beam_trees[source_beam_ix].current_sequence + current_token_choice
            )
            is_final = step == len(scores) - 1 or current_token_choice_ix == tokenizer.eos_token_id
            beam_trees[source_beam_ix].children[current_token_choice_ix] = BeamNode(
                current_token_ix=current_token_choice_ix,
                table=None,
                children={},
                current_sequence=current_sequence,
                cumulative_score=cumulative_score,
                total_score=cumulative_score / (step + 1 ** length_penalty),
                children_score_divider=((step + 2) ** length_penalty),
                is_final=is_final,
                is_selected_sequence=(
                    current_sequence.replace("<|endoftext|>", "")
                    in [el.replace("<|endoftext|>", "") for el in decoded_sequences]
                ),
            )

        # Swap all beams by descending cumul score, so that n°1 has the highest cumulative score, and so on
        beam_trees = [
            beam_trees[int(top_df_selected_filtered.iloc[beam_ix]["beam_index"])]
            for beam_ix in range(beams_to_keep)
        ]

        # Advance all beams by one token
        for beam_ix in range(beams_to_keep):
            current_token_choice_ix = top_df_selected_filtered.iloc[beam_ix]["token_index"]
            beam_trees[beam_ix] = beam_trees[beam_ix].children[current_token_choice_ix]

    return original_tree


def get_beam_search_html(
    input_text, number_steps, number_beams, length_penalty, num_return_sequences
):
    inputs = tokenizer([input_text], return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=number_steps,
        num_beams=number_beams,
        num_return_sequences=min(num_return_sequences, number_beams),
        return_dict_in_generate=True,
        length_penalty=length_penalty,
        output_scores=True,
        do_sample=False,
    )
    markdown = "The conclusive sequences are the ones that end in an `<|endoftext|>` token or at the end of generation."
    markdown += "\n\nThey are ranked by their scores, as given by the formula `score = cumulative_score / (output_length ** length_penalty)`.\n\n"
    markdown += "Only the top `num_beams` scoring sequences are returned: in the tree they are highlighted in **<span style='color:var(--secondary-500)!important'>blue</span>**."
    markdown += " The non-selected sequences are also shown in the tree, highlighted in **<span style='color:var(--primary-500)!important'>yellow</span>**."
    markdown += "\n#### <span style='color:var(--secondary-500)!important'>Output sequences:</span>"
    # Sequences are padded anyway so you can batch decode them
    decoded_sequences = tokenizer.batch_decode(outputs.sequences)

    if number_beams > 1:
        for i, sequence in enumerate(decoded_sequences):
            markdown += f"\n- Score `{outputs.sequences_scores[i]:.2f}`: `{clean(sequence.replace('<s> ', ''))}`"
    else:
        markdown += f"\n- `{clean(decoded_sequences[0].replace('<s> ', ''))}`"

    if number_beams > 1:
        original_tree = generate_beams(
            number_beams,
            input_text,
            outputs.scores[:],
            length_penalty,
            decoded_sequences,
        )
    else:
        original_tree = generate_beams(
            number_beams,
            input_text,
            outputs.scores,
            0,
            decoded_sequences,
        )
        
    html = generate_html(input_text, original_tree)
    return html, markdown


def change_num_return_sequences(n_beams):
    return gr.Slider(
        label="Number of sequences", minimum=1, maximum=n_beams, step=1, value=n_beams
    )


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.yellow,
        secondary_hue=gr.themes.colors.blue,
    ),
    css=STYLE,
) as demo:
    gr.Markdown(
        """# <span style='color:var(--primary-500)!important'>Beam Search Visualizer</span>

Play with the parameters below to understand how beam search decoding works!

Here's GPT2 doing beam search decoding for you.

#### <span style='color:var(--primary-500)!important'>Parameters:</span>
- **Sentence to decode from** (`inputs`): the input sequence to your decoder.
- **Number of steps** (`max_new_tokens`): the number of tokens to generate.
- **Number of beams** (`num_beams`): the number of beams to use.
- **Length penalty** (`length_penalty`): the length penalty to apply to outputs. `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
This parameter will not impact the beam search paths, but only influence the choice of sequences in the end towards longer or shorter sequences.
- **Number of return sequences** (`num_return_sequences`): the number of sequences to be returned at the end of generation. Should be `<= num_beams`.
"""
    )
    text = gr.Textbox(
        label="Sentence to decode from",
        value="Conclusion: thanks a lot. That's all for today",
    )
    with gr.Row():
        n_steps = gr.Slider(
            label="Number of steps", minimum=1, maximum=12, step=1, value=5
        )
        n_beams = gr.Slider(
            label="Number of beams", minimum=1, maximum=4, step=1, value=4
        )
        length_penalty = gr.Slider(
            label="Length penalty", minimum=-3, maximum=3, step=0.5, value=1
        )
        num_return_sequences = gr.Slider(
            label="Number of return sequences", minimum=1, maximum=4, step=1, value=3
        )

    n_beams.change(
        fn=change_num_return_sequences, inputs=n_beams, outputs=num_return_sequences
    )
    button = gr.Button()
    out_html = gr.Markdown()
    out_markdown = gr.Markdown()
    button.click(
        get_beam_search_html,
        inputs=[text, n_steps, n_beams, length_penalty, num_return_sequences],
        outputs=[out_html, out_markdown],
    )

demo.launch(share=False)
