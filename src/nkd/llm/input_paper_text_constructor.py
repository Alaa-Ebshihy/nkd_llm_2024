"""
Construct the paper input according to the type of the input
"""

from src.utils.io_util import *


def construct_paper_input_text(input_text_type, paper_annotation_path):
    """
    :param input_text_type: can be either "full" or "az"
    :param paper_annotation_path: can be the full paper parsed path or the az path
    :return:
    """
    if input_text_type == 'az':
        return construct_paper_input_text_az_labels(paper_annotation_path)
    return construct_paper_input_full_text(paper_annotation_path)


def construct_paper_input_full_text(paper_discourse_section_path):
    discourse_section_data = read_json(paper_discourse_section_path)
    paper_input_text = {'title': discourse_section_data['title'],
                        'sections': []}
    print(discourse_section_data['title'])
    section_text_map = {}
    for field in discourse_section_data['fields']:
        if field['annotation']['discourse_section']['section_id'] in {'BACKGROUND', 'RESULT', 'CONCLUSION'}:
            continue
        section_name = field['annotation']['discourse_section']['section_name']
        if section_name not in section_text_map:
            section_text_map[section_name] = []
        if ' '.join(field['text']).strip() != "":
            section_text_map[section_name].append(' '.join(field['text']))

    for section_name in section_text_map:
        paper_input_text['sections'].append({'section_title': section_name,
                                             'paragraphs': section_text_map[section_name]})
    return json.dumps(paper_input_text)


def construct_paper_input_text_az_labels(paper_az_annotation_path):
    az_annotation_data = read_json(paper_az_annotation_path)
    paper_input_text = {'title': az_annotation_data['title'],
                        'sections': []}
    for section in az_annotation_data['sections']:
        if len(section['selected_sentences']) == 0:
            continue
        paper_input_text['sections'].append({
            'section_title': section['section_name'],
            'paragraphs': section['selected_sentences']
        })
    return json.dumps(paper_input_text)
