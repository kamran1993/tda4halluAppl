line_0 = {
    "condition": "altered",
    "prompt": "taxi { request ( arrive by ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {"intent": "request", "domain": "taxi", "slot": "arrive by", "value": ""}
        ],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {"intent": "request", "domain": "taxi", "slot": "leave at", "value": ""},
            {"intent": "request", "domain": "taxi", "slot": "arrive by", "value": ""},
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "binary",
                "idx": 0,
                "dropped_act": {
                    "intent": "request",
                    "domain": "taxi",
                    "slot": "leave at",
                    "value": "",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "What time do you want to leave and what time do you want to arrive by?",
    "predictions": {"utterance": "when would you like to arrive by?"},
    "prediction": "when would you like to arrive by?",
}
line_1 = {
    "condition": "altered",
    "prompt": "taxi { inform ( type = blue honda ) @ book (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "taxi",
                "slot": "type",
                "value": "blue honda",
            }
        ],
        "binary": [{"intent": "book", "domain": "taxi", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "taxi",
                "slot": "type",
                "value": "blue honda",
            },
            {
                "intent": "inform",
                "domain": "taxi",
                "slot": "phone",
                "value": "07218068540",
            },
        ],
        "binary": [{"intent": "book", "domain": "taxi", "slot": "", "value": ""}],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 1,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "taxi",
                    "slot": "phone",
                    "value": "07218068540",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "Booking completed! your taxi will be blue honda Contact number is 07218068540",
    "predictions": {
        "utterance": "Your booking is complete. A blue honda will be picking you up."
    },
    "prediction": "Your booking is complete. A blue honda will be picking you up.",
}
line_2 = {
    "condition": "altered",
    "prompt": "general { reqmore (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {"field_drops": [], "injected_noise": [], "fallback_kept": []},
    "utterance": "You are welcome.  Is there anything else I can help you with today?",
    "predictions": {"utterance": "Is there anything else I can help you with today?"},
    "prediction": "Is there anything else I can help you with today?",
}
line_3 = {
    "condition": "altered",
    "prompt": "general { bye (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {"field_drops": [], "injected_noise": [], "fallback_kept": []},
    "utterance": "you too! thank you",
    "predictions": {"utterance": "Thank you for using our service. Have a great day!"},
    "prediction": "Thank you for using our service. Have a great day!",
}
line_4 = {
    "condition": "altered",
    "prompt": "restaurant { request ( food ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "food", "value": ""}
        ],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "nooffer",
                "domain": "restaurant",
                "slot": "name",
                "value": "Nusha",
            }
        ],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "food", "value": ""}
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "nooffer",
                    "domain": "restaurant",
                    "slot": "name",
                    "value": "Nusha",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "I don't seem to be finding anything called Nusha.  What type of food does the restaurant serve?",
    "predictions": {"utterance": "What type of food are you looking for?"},
    "prediction": "What type of food are you looking for?",
}
line_5 = {
    "condition": "altered",
    "prompt": "restaurant { request ( name ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "name", "value": ""}
        ],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "name",
                "value": "Nandos",
            }
        ],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "name", "value": ""}
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "restaurant",
                    "slot": "name",
                    "value": "Nandos",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "Could you double check that you've spelled the name correctly? The closest I can find is Nandos.",
    "predictions": {
        "utterance": "I can help you with that.  What is the name of the restaurant you are looking for?"
    },
    "prediction": "I can help you with that.  What is the name of the restaurant you are looking for?",
}
line_6 = {
    "condition": "altered",
    "prompt": "attraction { inform ( type = entertainment ; address = unit g6 ; address = cambridge leisure park ; address = clifton road ) } | general { reqmore (  ) } &",
    "dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "type",
                "value": "entertainment",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "unit g6",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "cambridge leisure park",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "clifton road",
            },
        ],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "type",
                "value": "entertainment",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "area",
                "value": "south",
            },
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "unit g6",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "cambridge leisure park",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "clifton road",
            },
        ],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "categorical",
                "idx": 1,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "attraction",
                    "slot": "area",
                    "value": "south",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "oh its okay. that is an entertainment type located in the south at unit g6, cambridge leisure park, clifton road. do you need their phone number?",
    "predictions": {
        "utterance": "It is an entertainment venue located at Unit G6, Cambridge Leisure Park, Clifton Road. Is there anything else I can help you with?"
    },
    "prediction": "It is an entertainment venue located at Unit G6, Cambridge Leisure Park, Clifton Road. Is there anything else I can help you with?",
}
line_7 = {
    "condition": "altered",
    "prompt": "attraction { inform ( postcode = cb17dy ; address = unit g6 ; address = cambridge leisure park ) } | general { reqmore (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "postcode",
                "value": "CB17DY",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "Unit g6",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "Cambridge Leisure Park",
            },
        ],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "postcode",
                "value": "CB17DY",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "Unit g6",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "Cambridge Leisure Park",
            },
            {
                "intent": "inform",
                "domain": "attraction",
                "slot": "address",
                "value": "Clifton road.",
            },
        ],
        "binary": [{"intent": "reqmore", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 3,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "attraction",
                    "slot": "address",
                    "value": "Clifton road.",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "Their address is Unit g6, Cambridge Leisure Park, Clifton road. The postcode is CB17DY. Is their anything else I can do for you?",
    "predictions": {
        "utterance": "The address is unit g6, cambridge leisure park, cb17dy. Is there anything else I can help you with?"
    },
    "prediction": "The address is unit g6, cambridge leisure park, cb17dy. Is there anything else I can help you with?",
}
line_8 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( area = centre ; choice = nine ) @ request ( price range ) } &",
    "dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "area",
                "value": "centre",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "choice",
                "value": "nine",
            }
        ],
        "binary": [
            {
                "intent": "request",
                "domain": "restaurant",
                "slot": "price range",
                "value": "",
            }
        ],
    },
    "original_dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "area",
                "value": "centre",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "food",
                "value": "Indian",
            },
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "choice",
                "value": "nine",
            },
        ],
        "binary": [
            {
                "intent": "request",
                "domain": "restaurant",
                "slot": "price range",
                "value": "",
            }
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "restaurant",
                    "slot": "food",
                    "value": "Indian",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "There are nine Indian restaurants in centre what price range do you want?",
    "predictions": {
        "utterance": "There are nine restaurants in the centre. What price range are you looking for?"
    },
    "prediction": "There are nine restaurants in the centre. What price range are you looking for?",
}
line_9 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( price range = expensive ; name = saffron brasserie ) } &",
    "dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "price range",
                "value": "expensive",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "name",
                "value": "saffron brasserie",
            }
        ],
        "binary": [],
    },
    "original_dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "price range",
                "value": "expensive",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "food",
                "value": "italian",
            },
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "name",
                "value": "saffron brasserie",
            },
        ],
        "binary": [],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "restaurant",
                    "slot": "food",
                    "value": "italian",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "saffron brasserie is an expensive restaurant that serves italian food",
    "predictions": {
        "utterance": "Saffron Brasserie is in the expensive price range. Would you like me to make a reservation for you?"
    },
    "prediction": "Saffron Brasserie is in the expensive price range. Would you like me to make a reservation for you?",
}
line_10 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( address = hills road city centre ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "address",
                "value": "Hills Road City Centre",
            }
        ],
        "binary": [],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "address",
                "value": "Hills Road City Centre",
            }
        ],
        "binary": [],
    },
    "alteration_meta": {"field_drops": [], "injected_noise": [], "fallback_kept": []},
    "utterance": "the address is Hills Road City Centre",
    "predictions": {"utterance": "The address is Hills Road City Centre."},
    "prediction": "The address is Hills Road City Centre.",
}
line_11 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( name = saffron brasserie ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "name",
                "value": "saffron brasserie",
            }
        ],
        "binary": [],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "food",
                "value": "Indian",
            },
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "name",
                "value": "saffron brasserie",
            },
        ],
        "binary": [],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "restaurant",
                    "slot": "food",
                    "value": "Indian",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "Yes, saffron brasserie is Indian food.",
    "predictions": {
        "utterance": "I have the saffron brasserie. Would you like me to make a reservation for you?"
    },
    "prediction": "I have the saffron brasserie. Would you like me to make a reservation for you?",
}
line_12 = {
    "condition": "altered",
    "prompt": "general { bye (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {"field_drops": [], "injected_noise": [], "fallback_kept": []},
    "utterance": "i hope i have been of help",
    "predictions": {"utterance": "Thank you for using our service. Have a great day."},
    "prediction": "Thank you for using our service. Have a great day.",
}
line_13 = {
    "condition": "altered",
    "prompt": "general { bye (  ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [],
        "binary": [{"intent": "bye", "domain": "general", "slot": "", "value": ""}],
    },
    "alteration_meta": {"field_drops": [], "injected_noise": [], "fallback_kept": []},
    "utterance": "I am glad to help. Enjoy your stay !",
    "predictions": {
        "utterance": "Thank you for contacting Cambridge TownInfo centre. Goodbye!"
    },
    "prediction": "Thank you for contacting Cambridge TownInfo centre. Goodbye!",
}
line_14 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( choice = many options ) @ request ( area ) } &",
    "dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "choice",
                "value": "many options",
            }
        ],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "area", "value": ""}
        ],
    },
    "original_dialogue_acts": {
        "categorical": [],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "choice",
                "value": "many options",
            }
        ],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "area", "value": ""},
            {"intent": "request", "domain": "restaurant", "slot": "food", "value": ""},
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "binary",
                "idx": 1,
                "dropped_act": {
                    "intent": "request",
                    "domain": "restaurant",
                    "slot": "food",
                    "value": "",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "I have many options available for you! Is there a certain area or cuisine that interests you?",
    "predictions": {
        "utterance": "There are many options. Is there a certain area you would like to dine in?"
    },
    "prediction": "There are many options. Is there a certain area you would like to dine in?",
}
line_15 = {
    "condition": "altered",
    "prompt": "restaurant { inform ( area = centre of town ) @ request ( food ) } &",
    "dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "area",
                "value": "centre of town",
            }
        ],
        "non-categorical": [],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "food", "value": ""}
        ],
    },
    "original_dialogue_acts": {
        "categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "area",
                "value": "centre of town",
            }
        ],
        "non-categorical": [
            {
                "intent": "inform",
                "domain": "restaurant",
                "slot": "choice",
                "value": "21",
            }
        ],
        "binary": [
            {"intent": "request", "domain": "restaurant", "slot": "food", "value": ""}
        ],
    },
    "alteration_meta": {
        "field_drops": [
            {
                "bucket": "non-categorical",
                "idx": 0,
                "dropped_act": {
                    "intent": "inform",
                    "domain": "restaurant",
                    "slot": "choice",
                    "value": "21",
                },
            }
        ],
        "injected_noise": [],
        "fallback_kept": [],
    },
    "utterance": "There are 21 restaurants available in the centre of town. How about a specific type of cuisine?",
    "predictions": {
        "utterance": "I can help you with that.  Is there a certain type of food you're looking for?  The centre of town has many options."
    },
    "prediction": "I can help you with that.  Is there a certain type of food you're looking for?  The centre of town has many options.",
}
