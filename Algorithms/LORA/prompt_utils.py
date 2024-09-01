### version1.0
# DATASET_TO_INSTRUCTION_MAPPING = {
#     "YELP-NEGATIVE": "Please change the sentiment of the following sentence to be more positive.",
#     "YELP-POSITIVE": "Please change the sentiment of the following sentence to be more negative.",
#     "GYAFC-INFORMAL": "Please rewrite the following sentence to be more formal.",
#     "GYAFC-FORMAL": "Please rewrite the following sentence to be more informal.",
#     "AMAZON-NEGATIVE": "Please change the sentiment of the following sentence to be more positive.",
#     "AMAZON-POSITIVE": "Please change the sentiment of the following sentence to be more negative.",
#     "GYAFC_EM_INFORMAL": "Please rewrite the following sentence to be more formal.",
#     "GYAFC_EM_FORMAL": "Please rewrite the following sentence to be more informal.",
#     "SHAKESPEARE-ORIGINAL": "Rewrite the following sentence, maintain the content and change the style of the sentence from Shakespeare English to modern English:",
#     "SHAKESPEARE-MODERN": "Rewrite the following sentence, maintain the content and change the style of the sentence from modern English to Shakespeare English:",
#     "PTB-REMOVAL": "Rewrite the following sentence, apply the semantic transfer to that sentence which is to remove adjectives and adverbs:",
#     "PTB-FUTURE": "Rewrite the following sentence,  apply the syntax transfer to that sentence which is to convert it into the future tense:",
# }


### version2.0
DATASET_TO_INSTRUCTION_MAPPING = {
    "YELP-NEGATIVE": "Rewrite the following sentence, maintain the content and change the sentiment of the sentence from negative to positive:",
    "YELP-POSITIVE": "Rewrite the following sentence, maintain the content and change the sentiment of the sentence from positive to negative:",
    "AMAZON-NEGATIVE": "Rewrite the following amazon comment, maintain the content and change the sentiment of the sentence from negative to positive:",
    "AMAZON-POSITIVE": "Rewrite the following amazon comment, maintain the content and change the sentiment of the sentence from positive to negative:",
    "GYAFC_FR_INFORMAL": "Change the style of the following sentence from informal to formal:",
    "GYAFC_FR_FORMAL": "Change the style of the following sentence from formal to informal:",
    "GYAFC_EM_INFORMAL": "Change the style of the following sentence from informal to formal:",
    "GYAFC_EM_FORMAL": "Change the style of the following sentence from formal to informal:",
    "SHAKESPEARE-ORIGINAL": "Change the style of the sentence from Shakespeare English to modern English:",
    "SHAKESPEARE-MODERN": "Change the style of the sentence from modern English to Shakespeare English:",
    "PTB-REMOVAL": "Remove adjectives and adverbs of the following sentence:",
    "PTB-FUTURE": "Convert the following sentence into the future tense:",
}
