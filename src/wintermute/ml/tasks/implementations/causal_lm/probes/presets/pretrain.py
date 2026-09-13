from __future__ import annotations

from wintermute.data.constants import TASK_PROBLEM_SOLVING
from wintermute.ml.tasks.implementations.causal_lm.probes import (
    JsonFields,
    MeanScore,
    Numeric,
    PassAtK,
    Probe,
    ProbeSampling,
    ProbeSuite,
)
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
)

from .builders import _chat, _chrf, _exact, _raw

_OBSERVABLE_SAMPLING = ProbeSampling(sample_count=1, greedy=True)

_METRIC_SAMPLING = ProbeSampling(
    sample_count=8,
    temperature=0.8,
    top_p=0.9,
    top_k=40,
)


def _direct_final(*, scalar: bool = False) -> AssistantPromptFormat:
    return AssistantPromptFormat(
        last_prompt_control_token=AssistantControlToken.FINAL,
        forced_task=TASK_PROBLEM_SOLVING,
        forced_format="scalar" if scalar else None,
    )

PRETRAIN_PROBE_SUITE = ProbeSuite(
    name="pretrain",
    probes=(
        # Factual knowledge and scientific explanation.
        _raw("factual/earth-orbit", "The Earth goes around the Sun because"),
        _raw(
            "narrative/rain-fourth-morning",
            "The rain started at noon and didn’t stop for three days. By the fourth morning,",
        ),
        _raw(
            "science/photosynthesis",
            "Photosynthesis is the process by which plants convert light energy into chemical energy. In most plants,",
        ),
        _raw(
            "science/correlation-causation",
            "The difference between correlation and causation is ",
        ),
        _raw(
            "knowledge/eiffel-location",
            "The Eiffel Tower is an iron lattice tower located in",
            _exact("Paris"),
        ),
        _chat("knowledge/capital-france", "What is the capital of France?", _exact("Paris")),
        _chat(
            "knowledge/capital-germany-fr",
            "Quelle est la capitale de l'Allemagne?",
            _exact("Berlin"),
        ),
        _chat("explanation/blue-sky", "Explain in one sentence why the sky appears blue."),

        # Arithmetic and structured answers.
        _raw(
            "arithmetic/apples-unit-price-raw",
            "I bought 4 apples for 48 USD. The price of an apple is",
            Numeric(expected=12, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/apples-unit-price",
            "I bought 4 apples for 48 USD total. How much is one apple? Answer with just the number.",
            Numeric(expected=12, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/oranges-total",
            "I bought 9 oranges. One orange costs 3 USD. How much will I pay ?",
            Numeric(expected=27, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/oranges-pears-total",
            "I bought 4 oranges and 3 pears. One orange costs 2 euros and one pear costs 5 euros. "
            "How much will I pay? Answer with just the number.",
            Numeric(expected=23, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/add-2-7",
            "What is 2 + 7? Answer with just a number.",
            Numeric(expected=9, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/add-14-17",
            "What is 14 + 17? Answer with just a number.",
            Numeric(expected=31, embedded_score=0.7),
        ),
        Probe(
            id="arithmetic/add-5-7",
            prompt="How much is 5 + 7? Answer with one number.",
            assistant_prompt_format=_direct_final(scalar=True),
            verifier=Numeric(expected=12, embedded_score=0.7),
        ),
        Probe(
            id="arithmetic/divide-72-12",
            prompt="How much is 72 / 12?",
            assistant_prompt_format=_direct_final(scalar=True),
            verifier=Numeric(expected=6, embedded_score=0.7),
        ),
        Probe(
            id="arithmetic/multiply-9-6",
            prompt="How much is 9 multiplied by 6? Answer with one number.",
            assistant_prompt_format=_direct_final(scalar=True),
            verifier=Numeric(expected=54, embedded_score=0.7),
        ),
        _chat(
            "arithmetic/add-9-19-fr",
            "Combien font 9 + 19?",
            Numeric(expected=28, embedded_score=0.7),
        ),
        _chat(
            "structured/marbles-fr-json",
            "Alice a 7 billes rouges et 5 billes vertes. Elle donne 3 billes rouges.\n"
            "Réponds avec un objet JSON valide contenant exactement les clés 'rouges' et 'vertes', sans aucun autre texte.",
            JsonFields(expected_fields={"rouges": (4,), "vertes": (5,)}),
        ),
        _chat(
            "structured/marbles-en-json",
            "Alice has 7 red marbles and 5 green marbles. She gives away 3 red marbles.\n"
            "Answer with a valid JSON object containing exactly the keys 'red' and 'green', with no additional text.",
            JsonFields(expected_fields={"red": (4,), "green": (5,)}),
        ),

        # Logical, relational, and sequence reasoning.
        _raw(
            "reasoning/cats-are-animals",
            "All cats are animals. Felix is a cat. Felix is",
            _exact("animal"),
        ),
        _chat("reasoning/transitive-comparison-symbol", "If A>B and B>C, then A ? C", _exact(">")),
        _chat(
            "reasoning/oldest-person",
            "Tom is older than Jim. Jim is older than Sam. Who is the oldest? Answer with just a name.",
            _exact("Tom"),
        ),
        _chat(
            "reasoning/youngest-person",
            "Tom is older than Jim. Jim is older than Sam. Who is the youngest? Answer with just a name.",
            _exact("Sam"),
        ),
        _raw("sequence/abcd", "A B C D A B C D A B ", _exact("C")),
        _raw(
            "reasoning/relative-size",
            "If Christopher is bigger than Alexander and Alexander is bigger than Marc, then Christopher is ",
            _exact("bigger than Marc"),
        ),
        _chat(
            "reasoning/syllogism-silent",
            "All falis are merks. No merk is silent. Noro is a fali.\n"
            "Is Noro silent? Answer only yes or no.",
            _exact("no"),
        ),

        # Commonsense, temporal, spatial, and coreference reasoning.
        _raw(
            "commonsense/pet-choice-fr",
            "Mathilde aime les chats et les chiens. Sa maman n'aime pas les chiens. Mathilde lui demande donc un ",
            _exact("chat"),
        ),
        _raw(
            "commonsense/pet-choice-en",
            "Mathilde likes cats and dogs. Her mother doesn't like dogs. Mathilde therefore asks her for a ",
            _exact("cat"),
        ),
        _raw(
            "commonsense/pet-choice-dog-fr",
            "Alexandre aime les chats et les chiens. Sa maman aime les chiens. Alexandre lui demande donc un ",
            _exact("chien"),
        ),
        _raw(
            "commonsense/fallen-cup-location",
            "The cup was on the table. It fell down. Where is it now?",
            _exact("floor"),
        ),
        _chat(
            "commonsense/fallen-cup-choice",
            "The cup was on the table. It fell down. Is it more likely on the floor or still on the table?\n"
            "Answer with one word.",
            _exact("floor"),
        ),
        _raw("commonsense/umbrella-reason", "Alexander gave Emma the umbrella because"),
        _raw(
            "reasoning/book-recipient",
            "Peter, Alexander, Emma and Christopher went to the bookstore. Emma and Peter left. Then, Christopher gave a book to ",
            _exact("Alexander"),
        ),
        _raw("procedural/make-tea", "To make a cup of tea, first"),
        _raw("sequence/weekdays", "Monday, Tuesday, Wednesday,", _exact("Thursday")),
        _raw(
            "coreference/book-recipient",
            "When Alexander and Emma went to the store, Alexander gave a book to ",
            _exact("Emma"),
        ),

        # Translation and bilingual generation.
        _raw(
            "translation/en-fr-past-perfect-raw",
            "The French translation of 'She had already left when the meeting began.' is ",
            _chrf(
                "Elle était déjà partie quand la réunion a commencé.",
                "Elle était déjà partie lorsque la réunion a commencé.",
            ),
        ),
        _chat(
            "translation/en-fr-numbers",
            "Translate the following text into French. Preserve all numbers exactly. "
            "Return only the translation.\n"
            "Text: The train leaves on March 12 at 7:45 p.m. and the ticket costs 38 euros.",
            _chrf(
                "Le train part le 12 mars à 7 h 45 du soir et le billet coûte 38 euros.",
                "Le train part le 12 mars à 19 h 45 et le billet coûte 38 euros.",
            ),
        ),
        _raw(
            "translation/en-fr-idiom-raw",
            "The natural French translation of "
            "'Sarah met Thomas near Central Park, but the weather was not her cup of tea.' is ",
            _chrf(
                "Sarah a rencontré Thomas près de Central Park, mais la météo n'était pas à son goût.",
                "Sarah a rencontré Thomas près de Central Park, mais le temps n'était pas à son goût.",
            ),
        ),
        _chat(
            "translation/en-fr-coreference",
            "Translate the following passage into French. Return only the translation.\n"
            "Text: Emma thanked Claire because Claire had found Emma's missing keys. "
            "She returned them before leaving.",
            _chrf(
                "Emma a remercié Claire parce que Claire avait retrouvé les clés perdues d'Emma. "
                "Elle les lui a rendues avant de partir.",
                "Emma a remercié Claire, car Claire avait retrouvé les clés qu'Emma avait perdues. "
                "Elle les lui a rendues avant de partir.",
            ),
        ),
        _raw(
            "translation/fr-en-past-progressive-raw",
            "La traduction anglaise de "
            "'Elle travaillait depuis deux heures quand son ordinateur est tombé en panne.' est ",
            _chrf(
                "She had been working for two hours when her computer broke down.",
                "She had been working for two hours when her computer crashed.",
            ),
        ),
        _chat(
            "translation/fr-en-numbers",
            "Traduis le texte suivant en anglais en conservant exactement tous les nombres. "
            "Réponds uniquement avec la traduction.\n"
            "Texte : Le colis de 12,5 kg arrivera le 4 novembre avant 16 h 30.",
            _chrf(
                "The 12.5 kg parcel will arrive on November 4 before 4:30 p.m.",
                "The 12.5 kg package will arrive on November 4 before 4:30 p.m.",
            ),
        ),
        _chat(
            "translation/fr-en-bridge",
            "Traduis le passage suivant en anglais. Réponds uniquement avec la traduction.\n"
            "Texte : La mairie a fermé le pont mardi matin. Les ingénieurs ont détecté une fissure, "
            "mais ils pensent pouvoir le rouvrir vendredi.",
            _chrf(
                "The town hall closed the bridge on Tuesday morning. Engineers detected a crack, "
                "but they think they can reopen it on Friday.",
                "The city hall closed the bridge on Tuesday morning. Engineers found a crack, "
                "but they think they can reopen it on Friday.",
            ),
        ),

        # Open-ended writing and text completion.
        _raw(
            "generation/focus-tips",
            "Here are five practical tips for staying focused while working from home:\n1.",
        ),
        _raw(
            "generation/dialogue-continuation",
            "“I didn’t mean to lie,” she said.\n“Then why did you?”\nShe looked away and",
        ),
        _raw("generation/old-door-continuation", "When I opened the old wooden door, I found"),
        _raw(
            "generation/train-platform-continuation",
            "He looked at the empty train platform and suddenly understood that",
        ),
        _chat(
            "generation/funny-cat-reply",
            "Write a short and funny reply to: 'I am late because my cat stole my sock.'",
        ),
        _chat("generation/science-fiction", "Tell me a short science fiction idea."),

        # Code completion.
        _raw("code/python-print-loop", "for i in range(5):\n    print(", _exact("i)")),
        _chat(
            "code/python-add",
            "Complete this Python function.\ndef add(a, b):\n    return",
            _exact("a + b"),
        ),
        _chat(
            "code/python-add-squares",
            "Complete this Python function.\ndef add_squares(a, b):\n    return",
        ),

        # Reading comprehension and summarization.
        _chat(
            "extraction/cat-json",
            "Read this short text: Mia has a cat named Luna. Luna sleeps on the sofa.\n"
            "Return a valid JSON object with exactly the keys 'animal', 'name', and 'place'.",
            JsonFields(
                expected_fields={
                    "animal": ("cat",),
                    "name": ("Luna",),
                    "place": ("sofa", "the sofa"),
                }
            ),
        ),
        _chat(
            "summarization/factory-shutdown",
            "A factory stopped production on Monday after a power outage. Power was restored on Tuesday morning, but safety checks delayed the restart until Wednesday. No employees were injured.\n"
            "Summarize the cause and duration of the shutdown in a single sentence.",
        ),
        _chat(
            "summarization/library-renovation",
            "On Thursday, the city library reopened after a two-week renovation that had required the building to close during the busiest part of the school holidays. The work repaired several leaks in the aging roof, refreshed the main reading hall, and created a new children's room with low shelves, cushions, and space for weekly storytelling sessions. Library director Elena Martin said the changes were designed to make the building more welcoming to families while protecting its historic collection from future water damage. To mark the reopening, the library will host free guided tours on Saturday morning and waive late-return fees until the end of the month. It will also remain open until 8 p.m. on weekdays, two hours later than its previous closing time.\n"
            "Summarize this text in one sentence.",
        ),
        _chat(
            "extraction/aurora-battery-json",
            "Dr. Maya Chen presented the Aurora battery project in Lyon on 14 May 2026. The prototype stores 40 percent more energy than the previous model and will be tested by three local bus companies.\n"
            "Extract the facts as a valid JSON object with exactly the keys 'person', 'project', 'city', 'date', 'improvement', and 'testers'. Return no other text.",
            JsonFields(
                expected_fields={
                    "person": ("Dr. Maya Chen", "Maya Chen"),
                    "project": ("Aurora battery project", "Aurora"),
                    "city": ("Lyon",),
                    "date": ("14 May 2026", "May 14, 2026"),
                    "improvement": ("40 percent more energy", "40% more energy"),
                    "testers": ("three local bus companies", "3 local bus companies"),
                }
            ),
        ),

        # Text paraphrase and controlled classification.
        _chat(
            "rewriting/meeting-postponed",
            "Rewrite the following sentence in different words without adding or removing any information. "
            "Return one sentence only.\n"
            "The meeting was postponed because the train strike prevented several participants from arriving.",
        ),
        _chat(
            "classification/customer-praise",
            "Classify the message as exactly one of: complaint, request, praise, cancellation.\n"
            "Message: The replacement arrived today and works perfectly. Thank you for resolving the issue so quickly.\n"
            "Return only the label.",
            _exact("praise"),
        ),
        _chat(
            "classification/science-fr",
            "Classe le texte dans exactement une catégorie parmi : politique, science, sport, économie.\n"
            "Texte : Des chercheurs ont observé une nouvelle exoplanète grâce au télescope spatial.\n"
            "Réponds uniquement avec la catégorie.",
            _exact("science"),
        ),
    ),
    observable_sampling=_OBSERVABLE_SAMPLING,
    metrics=(
        MeanScore(sampling=_METRIC_SAMPLING),
        PassAtK(k=1, threshold=1.0, sampling=_METRIC_SAMPLING),
        PassAtK(k=4, threshold=1.0, sampling=_METRIC_SAMPLING),
        PassAtK(
            name="lpass@4",
            k=4,
            threshold=0.5,
            sampling=_METRIC_SAMPLING,
        ),
    ),
)
