"""Explicit Hugging Face file selections shared by the xP3 instruction presets."""

_XP3_BASE_URL = "https://huggingface.co/datasets/bigscience/xP3/resolve/main"


def _urls_for(paths: list[str]) -> list[str]:
    return [f"{_XP3_BASE_URL}/{path}" for path in paths]

XP3_FR_TEXT_PROCESSING_FILES_BY_TASK = {
    "summary": [
        "fr/xp3_GEM_wiki_lingua_en_fr_train_article_summary_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_en_fr_train_rephrase_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_en_fr_train_summarize_above_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_en_fr_train_tldr_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_en_fr_train_write_abstract_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_article_summary_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_rephrase_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_summarize_above_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_tldr_en.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_write_abstract_en.jsonl",
        "fr/xp3_GEM_xlsum_french_train_docsummary.jsonl",
        "fr/xp3_GEM_xlsum_french_train_prevcontent.jsonl",
        "fr/xp3_GEM_xlsum_french_train_tldr.jsonl",
    ],
    "generate": [
        "fr/xp3_GEM_wiki_lingua_en_fr_train_xp3longchars.jsonl",
        "fr/xp3_GEM_wiki_lingua_en_fr_train_xp3longwritearticle.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_xp3longchars.jsonl",
        "fr/xp3_GEM_wiki_lingua_fr_train_xp3longwritearticle.jsonl",
        "fr/xp3_GEM_xlsum_french_train_goodtitle.jsonl",
        "fr/xp3_GEM_xlsum_french_train_xp3longcontinue.jsonl",
        "fr/xp3_GEM_xlsum_french_train_xp3longgenarticle.jsonl",
        "fr/xp3_GEM_xlsum_french_train_xp3longimaginearticle.jsonl",
        "fr/xp3_GEM_xlsum_french_train_xp3longrest.jsonl",
    ],
}


XP3_EN_REWRITE_FILES = [
    "en/xp3_paws-x_en_train_paraphrase-task.jsonl",
]


XP3_EN_TEXT_PROCESSING_FILES_BY_TASK = {
    "summary": [
        "en/xp3_GEM_wiki_lingua_en_train_article_summary_en.jsonl",
        "en/xp3_GEM_wiki_lingua_en_train_rephrase_en.jsonl",
        "en/xp3_GEM_wiki_lingua_en_train_summarize_above_en.jsonl",
        "en/xp3_GEM_wiki_lingua_en_train_tldr_en.jsonl",
        "en/xp3_GEM_wiki_lingua_en_train_write_abstract_en.jsonl",
    ],
    "qa": [
        "en/xp3_adversarial_qa_dbert_train_answer_the_following_q.jsonl",
        "en/xp3_adversarial_qa_dbert_train_based_on.jsonl",
        "en/xp3_adversarial_qa_dbert_train_question_context_answer.jsonl",
        "en/xp3_adversarial_qa_dbert_train_tell_what_it_is.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_answer_the_following_q.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_based_on.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_question_context_answer.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_tell_what_it_is.jsonl",
        "en/xp3_adversarial_qa_droberta_train_answer_the_following_q.jsonl",
        "en/xp3_adversarial_qa_droberta_train_based_on.jsonl",
        "en/xp3_adversarial_qa_droberta_train_question_context_answer.jsonl",
        "en/xp3_adversarial_qa_droberta_train_tell_what_it_is.jsonl",
        "en/xp3_super_glue_boolq_train_GPT-3_Style.jsonl",
        "en/xp3_super_glue_boolq_train_I_wonder….jsonl",
        "en/xp3_super_glue_boolq_train_after_reading.jsonl",
        "en/xp3_super_glue_boolq_train_based_on_the_following_passage.jsonl",
        "en/xp3_super_glue_boolq_train_based_on_the_previous_passage.jsonl",
        "en/xp3_super_glue_boolq_train_could_you_tell_me….jsonl",
        "en/xp3_super_glue_boolq_train_exam.jsonl",
        "en/xp3_super_glue_boolq_train_exercise.jsonl",
        "en/xp3_super_glue_boolq_train_valid_binary.jsonl",
        "en/xp3_super_glue_boolq_train_yes_no_question.jsonl",
    ],
    "generate": [
        "en/xp3_GEM_wiki_lingua_en_train_xp3longchars.jsonl",
        "en/xp3_GEM_wiki_lingua_en_train_xp3longwritearticle.jsonl",
        "en/xp3_adversarial_qa_dbert_train_generate_question.jsonl",
        "en/xp3_adversarial_qa_dbert_train_xp3longgeneratecontext.jsonl",
        "en/xp3_adversarial_qa_dbert_train_xp3longwritecontext.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_generate_question.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_xp3longgeneratecontext.jsonl",
        "en/xp3_adversarial_qa_dbidaf_train_xp3longwritecontext.jsonl",
        "en/xp3_adversarial_qa_droberta_train_generate_question.jsonl",
        "en/xp3_adversarial_qa_droberta_train_xp3longgeneratecontext.jsonl",
        "en/xp3_adversarial_qa_droberta_train_xp3longwritecontext.jsonl",
        "en/xp3_ag_news_None_train_xp3longimagine.jsonl",
        "en/xp3_super_glue_boolq_train_xp3longprovidetext.jsonl",
    ],
}

XP3_FR_TEXT_PROCESSING_URLS_BY_TASK = {
    task: _urls_for(paths)
    for task, paths in XP3_FR_TEXT_PROCESSING_FILES_BY_TASK.items()
}
XP3_EN_TEXT_PROCESSING_URLS_BY_TASK = {
    task: _urls_for(paths)
    for task, paths in XP3_EN_TEXT_PROCESSING_FILES_BY_TASK.items()
}
XP3_EN_REWRITE_URLS = _urls_for(XP3_EN_REWRITE_FILES)

# The original general-knowledge selection mixed five routing behaviors. Keep the
# upstream files partitioned here so each raw datasource and snapshot is unitary.
XP3_EN_GENERAL_FILES_BY_TASK = {
    "classification": [
        "en/xp3_wiki_qa_None_train_Decide_good_answer.jsonl",
        "en/xp3_wiki_qa_None_train_Is_This_True?.jsonl",
        "en/xp3_wiki_qa_None_train_automatic_system.jsonl",
        "en/xp3_wiki_qa_None_train_exercise.jsonl",
        "en/xp3_wiki_qa_None_train_found_on_google.jsonl",
    ],
    "generation": [
        "en/xp3_trivia_qa_unfiltered_train_guess_question.jsonl",
        "en/xp3_wiki_qa_None_train_Generate_Question_from_Topic.jsonl",
        "en/xp3_wiki_qa_None_train_Jeopardy_style.jsonl",
        "en/xp3_wiqa_None_train_xp3longfollows.jsonl",
    ],
    "information_extraction": [
        "en/xp3_wiki_qa_None_train_Topic_Prediction_-_Answer_Only.jsonl",
        "en/xp3_wiki_qa_None_train_Topic_Prediction_-_Question_Only.jsonl",
        "en/xp3_wiki_qa_None_train_Topic_Prediction_-_Question_and_Answer_Pair.jsonl",
    ],
    "knowledge_qa": [
        "en/xp3_trivia_qa_unfiltered_train_first_person_context.jsonl",
        "en/xp3_trivia_qa_unfiltered_train_formal_description.jsonl",
        "en/xp3_trivia_qa_unfiltered_train_question_answer.jsonl",
        "en/xp3_trivia_qa_unfiltered_train_question_with_instruction.jsonl",
        "en/xp3_web_questions_None_train_get_the_answer.jsonl",
        "en/xp3_web_questions_None_train_potential-correct-answer.jsonl",
        "en/xp3_web_questions_None_train_question-answer.jsonl",
        "en/xp3_web_questions_None_train_short_general_knowledge_q.jsonl",
        "en/xp3_web_questions_None_train_whats_the_answer.jsonl",
        "en/xp3_wiki_qa_None_train_Direct_Answer_to_Question.jsonl",
    ],
    "problem_solving": [
        "en/xp3_ai2_arc_ARC-Challenge_train_heres_a_problem.jsonl",
        "en/xp3_ai2_arc_ARC-Challenge_train_i_am_hesitating.jsonl",
        "en/xp3_ai2_arc_ARC-Challenge_train_multiple_choice.jsonl",
        "en/xp3_ai2_arc_ARC-Challenge_train_pick_false_options.jsonl",
        "en/xp3_ai2_arc_ARC-Challenge_train_pick_the_most_correct_option.jsonl",
        "en/xp3_ai2_arc_ARC-Challenge_train_qa_options.jsonl",
        "en/xp3_wiqa_None_train_does_the_supposed_perturbation_have_an_effect.jsonl",
        "en/xp3_wiqa_None_train_effect_with_label_answer.jsonl",
        "en/xp3_wiqa_None_train_effect_with_string_answer.jsonl",
        "en/xp3_wiqa_None_train_what_is_the_final_step_of_the_following_process.jsonl",
        "en/xp3_wiqa_None_train_what_is_the_missing_first_step.jsonl",
        "en/xp3_wiqa_None_train_what_might_be_the_first_step_of_the_process.jsonl",
        "en/xp3_wiqa_None_train_what_might_be_the_last_step_of_the_process.jsonl",
        "en/xp3_wiqa_None_train_which_of_the_following_is_the_supposed_perturbation.jsonl",
    ],
}

XP3_EN_GENERAL_URLS_BY_TASK = {
    task: _urls_for(paths) for task, paths in XP3_EN_GENERAL_FILES_BY_TASK.items()
}
