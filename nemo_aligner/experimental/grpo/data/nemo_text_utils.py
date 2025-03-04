def apply_nemo_chat_template(messages, chat_prompt_tokens, add_generation_prompt=True):
    def convert_messages(input_list):
        output_dict = {
            'system': '',
            'conversations': [],
            'mask': 'User',
            'type': 'VALUE_TO_TEXT',
        }

        # Extract the system message
        for msg in input_list:
            if msg['role'] == 'system':
                output_dict['system'] = msg['content']
                break  # Assuming only one system message

        # Build the conversations list
        for msg in input_list:
            if msg['role'] != 'system':
                conversation_entry = {
                    'from': msg['role'].capitalize(),  # Capitalize 'user' and 'assistant'
                    'value': msg['content'],
                    'label': None,
                }
                output_dict['conversations'].append(conversation_entry)

        return output_dict
    
    # Adapted: https://github.com/NVIDIA/NeMo/blob/9c1e7a2634c7e5bd8b6b1b0d8a6257d037241916/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py
    SYSTEM_TOKEN = "System"
    TYPE_INSTRUCTION = {
        'TEXT_TO_VALUE': "",
        'VALUE_TO_TEXT': '',
    }

    def response_value_formater(label, label_start, end_signal):
        if isinstance(label, str):
            return label_start + label + end_signal
        elif label is None:
            return ''
        else:
            raise ValueError(f'Unknown label type {type(label)}, only str type is supported')

    def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
        TURN_TOKEN = special_tokens['turn_start']
        END_SIGNAL = special_tokens['end_of_turn']
        LABEL_START = special_tokens['label_start']
        END_NAME_SIGNAL = special_tokens['end_of_name']

        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = ""
        conversation = header
        for i, sentence in enumerate(source):
            sentence_from = sentence["from"]
            role_token = TURN_TOKEN
            if gtype is None:
                sentence["value"] = (
                    BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
                )
            elif gtype == "VALUE_TO_TEXT":
                sentence["value"] = (
                    BEGIN_SIGNAL
                    + role_token
                    + sentence_from
                    + END_NAME_SIGNAL
                    + (
                        response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                        if 'label' in sentence
                        else ''
                    )
                    + sentence["value"]
                    + END_SIGNAL
                )
            elif gtype == "TEXT_TO_VALUE":
                sentence["value"] = (
                    BEGIN_SIGNAL
                    + role_token
                    + sentence_from
                    + END_NAME_SIGNAL
                    + sentence["value"]
                    + END_SIGNAL
                    + (
                        response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                        if 'label' in sentence
                        else ''
                    )
                )
            else:
                raise ValueError(
                    f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
                )
            conversation += sentence["value"]
            # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
            if sentence_from != mask_role and i == len(source) - 1:
                conversation += TURN_TOKEN
        return conversation

    def _get_header_conversation_type_mask_role(source, special_tokens):
        END_SIGNAL = special_tokens['end_of_turn']
        END_NAME_SIGNAL = special_tokens['end_of_name']

        data_type = None
        if 'type' in source:
            data_type = source['type']
            if data_type is not None:
                assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
        # add end signal and concatenate together
        conversation = source['system']
        if data_type is not None:
            if TYPE_INSTRUCTION[data_type] != '':
                conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
        mask_role = source.get('mask', 'User')
        header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
        conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
        return header, conversation, data_type, mask_role

    if add_generation_prompt:
        # Make a copy to prevent modifying the original messages
        messages = messages + [
            {'role': 'assistant', 'content': ''}
        ]  # adding trailing assistant message so that prompt ends with Assistant tag.
    special_tokens = chat_prompt_tokens
    nemo_source = convert_messages(messages)
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(
        nemo_source, special_tokens
    )
    if add_generation_prompt:
        len_strip = len(special_tokens['end_of_turn'] + special_tokens['turn_start'])
        conversation = conversation[:-len_strip]
    return conversation


# TODO: prob can delete, was just to validate the jinja chat_templates of some aligned models
def chat_prompt_tokens_to_hf_jinja_chat_template(chat_prompt_tokens: dict[str, str]) -> str:
    """
    Takes a dict of chat_prompt_tokens and spits out the jinja template for hf

    It only supports a specific format of the chat template. For example, it
    assumes 'user' and 'assistant' are the only roles and they are interleaved.
    """
    strip_repr_single_quotes = lambda x: x[1:-1] 
    end_of_name = strip_repr_single_quotes(repr(chat_prompt_tokens['end_of_name']))
    end_of_turn = strip_repr_single_quotes(repr(chat_prompt_tokens['end_of_turn']))
    jinja_template = (
        f"{{{{'{chat_prompt_tokens['system_turn_start']}System{end_of_name}'}}}}"
        "{% for message in messages %}{% if message['role'] == 'system' %}"
        "{{message['content'].strip()}}"
        "{% endif %}{% endfor %}"
        f"{{{{'{end_of_turn}'}}}}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        f"{{{{ '{chat_prompt_tokens['turn_start']}User{end_of_name}' + message['content'].strip() + '{end_of_turn}{chat_prompt_tokens['turn_start']}Assistant{end_of_name}' }}}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'].strip() }}"
        "{% endif %}{% endfor %}"
    )
    return jinja_template
