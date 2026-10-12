from lora_code.modeling import generate


def test_batched_left_padded_generation_matches_one_at_a_time(tokenizer, tiny_model):
    tiny_model.config.pad_token_id = tokenizer.pad_token_id
    instructions = ["short", "a somewhat longer instruction with more words", "mid length one"]
    batched = generate(tiny_model, tokenizer, instructions, max_new_tokens=8, batch_size=3)
    single = [generate(tiny_model, tokenizer, [i], max_new_tokens=8, batch_size=1)[0] for i in instructions]
    assert batched == single  # order restored and padding has no effect on greedy output
    assert tokenizer.padding_side == "right"  # restored afterwards
