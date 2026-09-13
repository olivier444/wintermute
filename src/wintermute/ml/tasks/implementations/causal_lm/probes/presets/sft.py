from __future__ import annotations

from wintermute.data.constants import TASK_INFORMATION_EXTRACTION
from wintermute.ml.tasks.implementations.causal_lm.probes import (
    AnyOf,
    ContainsAny,
    Exact,
    JsonFields,
    MeanScore,
    Numeric,
    PassAtK,
    Probe,
    ProbeSampling,
    ProbeSuite,
    PythonExpression,
    StartsWith,
    Verifier,
)
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
)

from .builders import _chat, _chrf, _exact, _raw

_GREEDY_SAMPLING = ProbeSampling(sample_count=1, greedy=True)

_RANDOM_SAMPLING = ProbeSampling(
    sample_count=16,
    temperature=0.8,
    top_p=0.9,
    top_k=40,
)


def _final_prompt(
    *,
    forced_task: str | None = None,
    forced_format: str | None = None,
) -> AssistantPromptFormat:
    return AssistantPromptFormat(
        last_prompt_control_token=AssistantControlToken.FINAL,
        forced_task=forced_task,
        forced_format=forced_format,
    )


def _number(id: str, prompt: str, verifier: Verifier) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_final_prompt(forced_format="number"),
        verifier=verifier,
    )


def _boolean(id: str, prompt: str, verifier: Verifier) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_final_prompt(forced_format="boolean"),
        verifier=verifier,
    )


def _python(id: str, prompt: str, verifier: Verifier) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_final_prompt(forced_format="python"),
        verifier=verifier,
    )


def _lookup_text(id: str, prompt: str, expected: str) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_final_prompt(forced_task=TASK_INFORMATION_EXTRACTION),
        verifier=Exact(expected),
    )


def _lookup_number(id: str, prompt: str, expected: int) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_final_prompt(
            forced_task=TASK_INFORMATION_EXTRACTION,
            forced_format="number",
        ),
        verifier=Numeric(expected=expected),
    )


SFT_PROBE_SUITE = ProbeSuite(
    name="sft",
    probes=(
        # Knowledge.
        _chat(
            "knowledge/earth-orbit",
            "Why does the Earth orbit the Sun? Answer in one sentence.",
            ContainsAny(("gravity", "gravitation", "gravitational", "pull")),
        ),
        _chat(
            "knowledge/photosynthesis-gas",
            "Which gas do plants absorb from the atmosphere during photosynthesis? "
            "Answer with the full English name only.",
            ContainsAny(("carbon dioxide", "CO2")),
        ),
        _chat(
            "knowledge/berlin-wall-year",
            "In what year did the Berlin Wall fall? Answer with the four-digit year only.",
            Numeric(expected=1989),
        ),
        _chat(
            "knowledge/h2o-common-name",
            "What is H2O commonly called? Answer with one word only.",
            _exact("water"),
        ),
        _chat(
            "knowledge/guernica-painter",
            "Who painted Guernica? Answer with the artist's full name only.",
            ContainsAny(("Picasso", "Pablo Picasso")),
        ),
        _raw(
            "knowledge/eiffel-location",
            "The Eiffel Tower is an iron lattice tower located in",
            _exact("Paris"),
        ),
        _chat(
            "knowledge/capital-france",
            "What is the capital of France?",
            _exact("Paris")),
        _chat(
            "knowledge/capital-germany-fr",
            "Quelle est la capitale de l'Allemagne?",
            _exact("Berlin"),
        ),
        _number(
            "knowledge/red-planet-qcm",
            "Which planet is known as the Red Planet?\n"
            "1. Mars\n"
            "2. Venus\n"
            "3. Saturn\n"
            "Return only the candidate number.",
            Numeric(expected=1),
        ),
        _number(
            "knowledge/largest-ocean-qcm",
            "Which is the largest ocean on Earth?\n"
            "1. Atlantic Ocean\n"
            "2. Indian Ocean\n"
            "3. Pacific Ocean\n"
            "Return only the candidate number.",
            Numeric(expected=3),
        ),

        # Problem solving.
        _raw(
            "problem/apples-unit-price-raw",
            "I bought 4 apples for 48 USD. The price of an apple is",
            Numeric(expected=12, embedded_score=0.7),
        ),
        _chat(
            "problem/apples-unit-price",
            "I bought 4 apples for 48 USD total. How much is one apple? Answer with just the number.",
            Numeric(expected=12, embedded_score=0.7),
        ),
        _chat(
            "problem/oranges-total",
            "I bought 9 oranges. One orange costs 3 USD. How much will I pay ?",
            Numeric(expected=27, embedded_score=0.7),
        ),
        _chat(
            "problem/oranges-pears-total",
            "I bought 4 oranges and 3 pears. One orange costs 2 euros and one pear costs 5 euros. "
            "How much will I pay? Answer with just the number.",
            Numeric(expected=23, embedded_score=0.7),
        ),
        _chat(
            "problem/books-change",
            "Mia has 12 books. She buys 5 more and gives 3 away. "
            "How many books does she have now? Answer with just the number.",
            Numeric(expected=14),
        ),
        _chat(
            "problem/equal-sharing",
            "Twenty candies are shared equally among four children. "
            "How many candies does each child receive? Answer with just the number.",
            Numeric(expected=5),
        ),
        _number(
            "problem/bus-passengers",
            "A bus has 28 passengers. At the next stop, 9 passengers leave and 4 get on. "
            "How many passengers are now on the bus? Return only the number.",
            Numeric(expected=23),
        ),
        _number(
            "problem/recipe-flour",
            "A recipe for 4 people uses 3 cups of flour. How many cups are needed for 8 people? "
            "Return only the number.",
            Numeric(expected=6),
        ),
        _number(
            "problem/bookshelves-fr",
            "Une bibliothèque range 8 livres sur chacune de ses 5 étagères. "
            "Combien de livres sont rangés au total ? Réponds uniquement avec le nombre.",
            Numeric(expected=40),
        ),

        # Arithmetic.
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
        _number(
            "arithmetic/add-5-7",
            "How much is 5 + 7? Answer with one number.",
            Numeric(expected=12),
        ),
        _number(
            "arithmetic/divide-72-12",
            "How much is 72 / 12?",
            Numeric(expected=6),
        ),
        _number(
            "arithmetic/multiply-9-6",
            "How much is 9 multiplied by 6? Answer with one number.",
            Numeric(expected=54),
        ),
        _chat(
            "arithmetic/add-9-19-fr",
            "Combien font 9 + 19?",
            Numeric(expected=28, embedded_score=0.7),
        ),

        # Structured answers.
        _chat(
            "structured/marbles-fr-json",
            "Alice a 7 billes rouges et 5 billes vertes. Elle donne 3 billes rouges.\n"
            "Réponds avec un objet JSON valide contenant exactement les clés 'rouges' et 'vertes', sans aucun autre texte.",
            JsonFields(
                expected_fields={"rouges": (4,), "vertes": (5,)},
                allow_comments=True,
            ),
        ),
        _chat(
            "structured/marbles-en-json",
            "Alice has 7 red marbles and 5 green marbles. She gives away 3 red marbles.\n"
            "Answer with a valid JSON object containing exactly the keys 'red' and 'green', with no additional text.",
            JsonFields(
                expected_fields={"red": (4,), "green": (5,)},
                allow_comments=True,
            ),
        ),
        _chat(
            "structured/appointment-json",
            "Eva's appointment is on Tuesday at 14:30 in room B.\n"
            "Return a valid JSON object with exactly the keys 'person', 'day', 'time', and 'room'.",
            JsonFields(
                expected_fields={
                    "person": ("Eva",),
                    "day": ("Tuesday",),
                    "time": ("14:30",),
                    "room": ("B", "room B"),
                },
                allow_comments=True,
            ),
        ),
        _chat(
            "structured/product-json",
            "The notebook costs 7 euros and is in stock.\n"
            "Return a valid JSON object with exactly the keys 'item', 'price', and 'in_stock'.",
            JsonFields(
                expected_fields={
                    "item": ("notebook",),
                    "price": (7,),
                    "in_stock": (True,),
                },
                allow_comments=True,
            ),
        ),
        _chat(
            "structured/cat-json",
            "Read this short text: Mia has a cat named Luna. Luna sleeps on the sofa.\n"
            "Return a valid JSON object with exactly the keys 'animal', 'name', and 'place'.",
            JsonFields(
                expected_fields={
                    "animal": ("cat",),
                    "name": ("Luna",),
                    "place": ("sofa", "the sofa"),
                },
                allow_comments=True,
            ),
        ),
        _chat(
            "structured/aurora-battery-json",
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
                },
                allow_comments=True,
            ),
        ),
        _number(
            "structured/museum-saturday-visitors",
            "Read the report and extract the requested number.\n"
            "Report: The museum welcomed 243 visitors on Friday, 302 on Saturday and 187 on Sunday. "
            "Ticket sales totaled 1,290 euros over the weekend.\n"
            "How many visitors came to the museum on Saturday? Return only the number.",
            Numeric(expected=302),
        ),
        _number(
            "structured/library-tuesday-loans",
            "Read the report and extract the requested number.\n"
            "Report: The library lent 46 books on Monday, 58 on Tuesday, and 41 on Wednesday. "
            "Late-return fees totaled 75 euros over those three days.\n"
            "How many books did the library lend on Tuesday? Return only the number.",
            Numeric(expected=58),
        ),
        _number(
            "structured/simple-lookup-1",
            "1. Pierre\n"
            "2. John\n"
            "3. Martin\n"
            "4. Oliver\n"
            "5. Luther\n"
            "6. Christopher\n"
            "7. Paul\n"
            "What number is Luther? "
            "Return only the candidate number.",
            Numeric(expected=5),
        ),
        _number(
            "structured/simple-lookup-2",
            "1. Tokyo\n"
            "2. Nairobi\n"
            "3. Lisbon\n"
            "4. Montreal\n"
            "5. Seoul\n"
            "6. Lima\n"
            "7. Oslo\n"
            "What number is Lisbon? "
            "Return only the candidate number.",
            Numeric(expected=3),
        ),

        # Lookup and retrieval primitives.
        _lookup_text(
            "lookup-key-to-value/numeric-key-position-2-en",
            "Use the following lookup table.\n"
            "14. cedar\n"
            "81. glacier\n"
            "33. copper\n"
            "67. meadow\n"
            "Which value is associated with key '81'? Return only the value.",
            "glacier",
        ),
        _lookup_text(
            "lookup-key-to-value/letter-key-position-4-fr",
            "Utilise la table de correspondance suivante.\n"
            "Q) rivière\n"
            "B) planète\n"
            "M) violon\n"
            "T) lanterne\n"
            "F) montagne\n"
            "Quelle valeur est associée à la clé « T » ? Réponds uniquement avec la valeur.",
            "lanterne",
        ),
        _lookup_text(
            "lookup-key-to-value/id-key-position-6-en",
            "Refer to the mapping below.\n"
            "K7 -> amber\n"
            "R4 -> harbor\n"
            "D9 -> velvet\n"
            "P2 -> orchid\n"
            "X5 -> marble\n"
            "N8 -> comet\n"
            "H3 -> willow\n"
            "Look up key 'N8'. Answer with its value only.",
            "comet",
        ),
        _lookup_number(
            "lookup-value-to-key/numeric-key-position-2-fr",
            "Consulte la table ci-dessous.\n"
            "42. forêt\n"
            "73. lune\n"
            "19. cuivre\n"
            "88. rivière\n"
            "Quelle clé est associée à la valeur « lune » ? Réponds uniquement avec la clé.",
            73,
        ),
        _lookup_text(
            "lookup-value-to-key/letter-key-position-4-en",
            "Read these entries carefully.\n"
            "W) lantern\n"
            "C) ocean\n"
            "J) maple\n"
            "R) silver\n"
            "A) valley\n"
            "Find the label for 'silver'. Answer with the label only.",
            "R",
        ),
        _lookup_text(
            "lookup-value-to-key/id-key-position-6-fr",
            "Utilise la table de correspondance suivante.\n"
            "B14 -> jardin\n"
            "M72 -> soleil\n"
            "Q31 -> fenêtre\n"
            "V48 -> océan\n"
            "D25 -> village\n"
            "K63 -> planète\n"
            "T91 -> forêt\n"
            "Quelle clé pointe vers « planète » ? Donne seulement la clé.",
            "K63",
        ),
        _lookup_text(
            "lookup-position-to-value/numeric-label-position-2-en",
            "Inspect the ordered entries.\n"
            "1. quartz\n"
            "2. saffron\n"
            "3. harbor\n"
            "4. willow\n"
            "Counting from the top starting at 1, what value is at position 2? Return only the value.",
            "saffron",
        ),
        _lookup_text(
            "lookup-position-to-value/letter-label-position-4-fr",
            "Examine les entrées ordonnées.\n"
            "A) rivière\n"
            "B) cuivre\n"
            "C) montagne\n"
            "D) fenêtre\n"
            "E) lanterne\n"
            "Quelle valeur occupe la position 4 ? Donne uniquement la valeur.",
            "fenêtre",
        ),
        _lookup_text(
            "lookup-position-to-value/id-label-position-6-en",
            "Read these entries carefully.\n"
            "H17 -> amber\n"
            "Q62 -> forest\n"
            "B29 -> copper\n"
            "N84 -> ocean\n"
            "R35 -> velvet\n"
            "K71 -> comet\n"
            "M46 -> garden\n"
            "Read top to bottom. What occupies position 6? Give only the value.",
            "comet",
        ),
        _lookup_number(
            "lookup-value-to-position/numeric-label-position-2-fr",
            "Lis attentivement les entrées suivantes.\n"
            "1. forêt\n"
            "2. lune\n"
            "3. rivière\n"
            "4. jardin\n"
            "En comptant depuis le haut à partir de 1, quelle est la position de « lune » ? "
            "Réponds uniquement avec le nombre.",
            2,
        ),
        _lookup_number(
            "lookup-value-to-position/letter-label-position-4-en",
            "Inspect the ordered entries.\n"
            "A) cedar\n"
            "B) glacier\n"
            "C) copper\n"
            "D) meadow\n"
            "E) lantern\n"
            "At which position does 'meadow' appear? Answer with the numeric position only.",
            4,
        ),
        _lookup_number(
            "lookup-value-to-position/id-label-position-6-fr",
            "Consulte la table ci-dessous.\n"
            "P18 -> ambre\n"
            "F62 -> port\n"
            "K39 -> velours\n"
            "T85 -> orchidée\n"
            "C27 -> marbre\n"
            "R74 -> comète\n"
            "M41 -> saule\n"
            "Lis de haut en bas. Donne la position de « comète » sous forme de nombre uniquement.",
            6,
        ),
        _number(
            "structured/odd-gender-out-1",
            "1. Ethan\n"
            "2. Lucas\n"
            "3. Noah\n"
            "4. Emma\n"
            "5. Christopher\n"
            "6. Henry\n"
            "7. Jack\n"
            "Which person is not a boy? "
            "Return only the candidate number.",
            Numeric(expected=4),
        ),

        # Reasoning.
        _chat(
            "reasoning/cats-are-animals",
            "All cats are animals. Felix is a cat. Felix is",
            _exact("animal"),
        ),
        _chat(
            "reasoning/transitive-comparison-symbol",
            "If A>B and B>C, then A ? C",
            AnyOf((Exact(">"), Exact("A>C"), Exact("A>B>C"))),
        ),
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
        _chat(
            "reasoning/relative-size",
            "If Christopher is bigger than Alexander and Alexander is bigger than Marc, then Christopher is ",
            AnyOf((_exact("bigger than Marc"),_exact("the biggest"))),
        ),
        _chat(
            "reasoning/syllogism-silent",
            "All falis are merks. No merk is silent. Noro is a fali.\n"
            "Is Noro silent? Answer only yes or no.",
            _exact("no"),
        ),
        _boolean(
            "reasoning/syllogism-green-boolean",
            "All ravis are tepons. All tepons are green. Mila is a ravi.\n"
            "Is Mila green? Answer only yes or no.",
            Exact("yes"),
        ),
        _boolean(
            "reasoning/left-of-transitive-boolean",
            "Jade is standing to the left of Omar. Omar is standing to the left of Nina.\n"
            "Is Nina to the left of Jade? Answer only yes or no.",
            Exact("no"),
        ),
        _raw(
            "reasoning/book-recipient-simple",
            "When Alexander and Emma went to the store, Alexander gave a book to ",
            _exact("Emma"),
        ),
        _chat(
            "reasoning/book-recipient-hard",
            "Peter, Alexander, Emma and Christopher went to the bookstore. Emma and Peter left. Then, Christopher gave a book to ",
            _exact("Alexander"),
        ),

        # Sequence completion.
        _number(
            "sequence/1234",
            "Continue the sequence: 2 3 4 1 2 3 4 1 2. Answer with the next number only.",
            Numeric(expected=3),
        ),
        _chat(
            "sequence/weekdays",
            "What is the next item of the following sequence: Monday, Tuesday, Wednesday,",
            _exact("Thursday")
        ),
        _number(
            "sequence/even-numbers",
            "What is the next number in this sequence: 2, 4, 6, 8? Answer with just the number.",
            Numeric(expected=10),
        ),
        _number(
            "sequence/multiples-of-three",
            "What is the next number in this sequence: 3, 6, 9, 12? Answer with just the number.",
            Numeric(expected=15),
        ),
        _number(
            "sequence/alternating-colors",
            "What comes next in this sequence: red, blue, red, blue, red?\n"
            "1. green\n"
            "2. blue\n"
            "3. red\n"
            "4. yellow\n"
            "Return only the candidate number.",
            Numeric(expected=2),
        ),

        # Commonsense.
        _chat(
            "commonsense/pet-choice-fr",
            "Mathilde aime les chats et les chiens. Sa maman n'aime pas les chiens. Mathilde lui demande donc un ",
            _exact("chat"),
        ),
        _chat(
            "commonsense/pet-choice-en",
            "Mathilde likes cats and dogs. Her mother doesn't like dogs. Mathilde therefore asks her for a ",
            _exact("cat"),
        ),
        _chat(
            "commonsense/pet-choice-dog-fr",
            "Alexandre aime les chats et les chiens. Sa maman accepte les chiens mais refuse les chats. "
            "Quel animal Alexandre doit-il lui demander ? Réponds uniquement par « chat » ou « chien ».",
            Exact("chien"),
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
        _number(
            "commonsense/ice-cube-sun-qcm",
            "An ice cube is left outside on a warm, sunny day. What will most likely happen?\n"
            "1. It will melt.\n"
            "2. It will grow larger.\n"
            "3. It will turn into metal.\n"
            "Return only the candidate number.",
            Numeric(expected=1),
        ),
        _number(
            "commonsense/rain-item-fr-qcm",
            "Il pleut et Nora doit marcher dehors. Quel objet devrait-elle prendre ?\n"
            "1. Un oreiller\n"
            "2. Un parapluie\n"
            "3. Une fourchette\n"
            "Réponds uniquement avec le numéro du candidat.",
            Numeric(expected=2),
        ),

        # Translation.
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
        _chat(
            "translation/fr-en-past-progressive",
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
        _number(
            "translation/en-fr-present-qcm",
            "Choose the best French translation of the following sentence.\n"
            "Text: The children are playing in the garden while their father prepares dinner.\n"
            "1. Les enfants ont joué dans le jardin pendant que leur père préparait le dîner.\n"
            "2. Les enfants jouent dans la cuisine pendant que leur père regarde le jardin.\n"
            "3. Les enfants jouent dans le jardin pendant que leur père prépare le dîner.\n"
            "Return only the candidate number.",
            Numeric(expected=3),
        ),
        _number(
            "translation/fr-en-hours-qcm",
            "Choisis la meilleure traduction anglaise de la phrase suivante.\n"
            "Texte : Le musée ouvre à 9 h 30 et ferme à 18 h, sauf le lundi.\n"
            "1. The museum opens at 9:30 a.m. and closes at 6 p.m., except on Mondays.\n"
            "2. The museum opens at 9:30 a.m. and closes at 6 p.m. only on Mondays.\n"
            "3. The museum opens at 6:30 a.m. and closes at 9 p.m., including Mondays.\n"
            "Réponds uniquement avec le numéro du candidat.",
            Numeric(expected=1),
        ),

        # Generation.
        _raw(
            "generation/focus-tips",
            "Here are five practical tips for staying focused while working from home:\n1.",
        ),
        _raw(
            "generation/dialogue-continuation",
            "“I didn’t mean to lie,” she said.\n“Then why did you?”\nShe looked away and",
        ),
        _raw(
            "generation/old-door-continuation",
            "When I opened the old wooden door, I found"
        ),
        _raw(
            "generation/train-platform-continuation",
            "He looked at the empty train platform and suddenly understood that",
        ),
        _chat(
            "generation/funny-cat-reply",
            "Write a short and funny reply to: 'I am late because my cat stole my sock.'",
        ),
        _chat(
            "generation/science-fiction",
            "Tell me a short science fiction idea."
        ),
        _raw(
            "generation/rain-fourth-morning",
            "The rain started at noon and didn’t stop for three days. By the fourth morning,",
        ),
        _number(
            "generation/qcm-river-slogan",
            "Choose the slogan that contains exactly six words, includes the word 'river', and has "
            "no exclamation mark.\n"
            "1. Protect our beautiful river starting today!\n"
            "2. Together we keep the river clean\n"
            "3. Together we keep every park clean\n"
            "Return only the candidate number.",
            Numeric(expected=2),
        ),
        _number(
            "generation/qcm-billing-support",
            "A customer reports a duplicate charge. Choose the reply that apologizes, acknowledges "
            "the problem, and promises to investigate without claiming it is already resolved.\n"
            "1. I'm sorry about the duplicate charge; I'll review your account and update you shortly.\n"
            "2. The duplicate charge was refunded today, so there is nothing further for us to "
            "investigate.\n"
            "3. Duplicate charges are covered in our billing policy, which you should read before "
            "contacting support.\n"
            "Return only the candidate number.",
            Numeric(expected=1),
        ),
        _number(
            "generation/qcm-plant-continuation",
            "Story: Noah watered his tomato plant every morning, but its leaves kept wilting. He "
            "checked the pot and discovered that the drainage hole was blocked.\n"
            "Choose the most coherent next sentence.\n"
            "1. He painted the outside of the pot blue to make it look brighter.\n"
            "2. He poured in more water every hour without clearing the blocked drainage hole.\n"
            "3. He cleared the hole so excess water could drain away.\n"
            "Return only the candidate number.",
            Numeric(expected=3),
        ),
        _number(
            "generation/qcm-eclipse-explanation",
            "Choose the best one-sentence explanation of a solar eclipse for an eight-year-old. It "
            "must be simple and factually accurate.\n"
            "1. A solar eclipse is an occultation produced by syzygy among the Sun, Moon, and Earth "
            "along their orbital nodes.\n"
            "2. A solar eclipse happens when the Moon moves between Earth and the Sun and blocks some "
            "sunlight.\n"
            "3. A solar eclipse happens when the Sun turns off its light for a few minutes.\n"
            "Return only the candidate number.",
            Numeric(expected=2),
        ),
        _number(
            "generation/qcm-neighborhood-invitation-fr",
            "Choisis l'invitation amicale qui indique tous les éléments demandés : nettoyage du "
            "quartier, samedi à 9 heures, rendez-vous sur la place de la mairie et gants à apporter.\n"
            "1. Rejoignez-nous samedi à 9 heures sur la place de la mairie pour nettoyer le quartier ; "
            "pensez à apporter vos gants !\n"
            "2. Le nettoyage du quartier aura lieu dimanche à 14 heures au parc central ; des sacs, "
            "des gants et des outils seront fournis à tous les participants.\n"
            "3. Une opération de nettoyage est prévue prochainement dans le quartier.\n"
            "Réponds uniquement avec le numéro du candidat.",
            Numeric(expected=1),
        ),
        _number(
            "generation/qcm-polite-refusal-fr",
            "Tu ne peux pas participer à une réunion vendredi. Choisis la réponse qui refuse poliment "
            "et propose lundi comme solution de remplacement.\n"
            "1. Vendredi est impossible pour moi, et je ne souhaite ni déplacer la réunion ni proposer "
            "une autre date pour la remplacer.\n"
            "2. Vendredi me convient parfaitement, à bientôt.\n"
            "3. Merci pour l'invitation. Je ne suis pas disponible vendredi, mais je pourrais participer "
            "lundi si cela vous convient.\n"
            "Réponds uniquement avec le numéro du candidat.",
            Numeric(expected=3),
        ),
        _number(
            "generation/qcm-vegetarian-soup-introduction",
            "Choose the warm one-sentence introduction for a vegetarian carrot-and-lentil soup. It "
            "must not make medical claims.\n"
            "1. This carrot-and-lentil soup cures colds immediately and guarantees strong health "
            "throughout the entire winter.\n"
            "2. Cozy carrots and lentils come together in this comforting vegetarian soup.\n"
            "3. Chicken and bacon make this carrot-and-lentil soup especially rich, smoky, and "
            "satisfying.\n"
            "Return only the candidate number.",
            Numeric(expected=2),
        ),

        # Code.
        _raw(
            "code/python-print-loop",
            "# Print the numbers 0 through 4, one per line.\n"
            "for i in range(5):\n"
            "    print(",
            StartsWith("i)"),
        ),
        _chat(
            "code/python-add",
            "Complete this Python function.\ndef add(a, b):\n    return",
            _exact("a + b"),
        ),
        _python(
            "code/python-analyze-numbers",
            "Write a complete multi-line Python function named `analyze_numbers` that takes a "
            "list named `numbers` and returns a dictionary with exactly these keys: `count` for "
            "the number of items, `has_negative` indicating whether any item is negative, and "
            "`all_even` indicating whether every item is even. Return only valid Python code.",
            PythonExpression(
                cases=(
                    (
                        {"numbers": []},
                        {"count": 0, "has_negative": False, "all_even": True},
                    ),
                    (
                        {"numbers": [2, 4, 6]},
                        {"count": 3, "has_negative": False, "all_even": True},
                    ),
                    (
                        {"numbers": [-2, 0, 3]},
                        {"count": 3, "has_negative": True, "all_even": False},
                    ),
                    (
                        {"numbers": [1, 3]},
                        {"count": 2, "has_negative": False, "all_even": False},
                    ),
                ),
                allow_comments=True,
            ),
        ),
        _python(
            "code/python-filter-even",
            "Complete this Python function so it returns the even numbers in their original order. "
            "Return only the missing expression.\n"
            "def even_numbers(numbers):\n"
            "    return",
            PythonExpression(
                cases=(
                    ({"numbers": []}, []),
                    ({"numbers": [1, 2, 3, 4]}, [2, 4]),
                    ({"numbers": [-3, -2, -1, 0]}, [-2, 0]),
                    ({"numbers": [2, 2, 5]}, [2, 2]),
                ),
                allow_comments=True,
            ),
        ),
        _python(
            "code/python-reverse-string",
            "Complete this Python function so it returns the text in reverse order. "
            "Return only the missing expression.\n"
            "def reverse_text(text):\n"
            "    return",
            PythonExpression(
                cases=(
                    ({"text": ""}, ""),
                    ({"text": "abc"}, "cba"),
                    ({"text": "Wintermute"}, "etumretniW"),
                    ({"text": "tea time"}, "emit aet"),
                ),
                allow_comments=True,
            ),
        ),
        _python(
            "code/python-word-lengths",
            "Complete this Python function so it maps each word to its length. "
            "Return only the missing expression.\n"
            "def word_lengths(words):\n"
            "    return",
            PythonExpression(
                cases=(
                    ({"words": []}, {}),
                    ({"words": ["tea", "AI", ""]}, {"tea": 3, "AI": 2, "": 0}),
                    ({"words": ["winter", "mute"]}, {"winter": 6, "mute": 4}),
                ),
                allow_comments=True,
            ),
        ),
        _python(
            "code/python-config-port",
            "Complete this Python function so it returns `config['port']`, or 8080 when the key is absent. "
            "Return only the missing expression.\n"
            "def config_port(config):\n"
            "    return",
            PythonExpression(
                cases=(
                    ({"config": {}}, 8080),
                    ({"config": {"port": 5432}}, 5432),
                    ({"config": {"port": None}}, None),
                    ({"config": {"host": "localhost"}}, 8080),
                ),
                allow_comments=True,
            ),
        ),
        _python(
            "code/python-any-negative",
            "Complete this Python function so it returns whether the input contains a negative number. "
            "Return only the missing expression.\n"
            "def has_negative(numbers):\n"
            "    return",
            PythonExpression(
                cases=(
                    ({"numbers": []}, False),
                    ({"numbers": [0, 2, 7]}, False),
                    ({"numbers": [0, -1, 2]}, True),
                    ({"numbers": [-3]}, True),
                ),
                allow_comments=True,
            ),
        ),

        # Summarization.
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
        _number(
            id="summarization/selection-pool-reopening",
            prompt=(
                "Source: The community pool closed on Monday because its water pump broke. The pump "
                "was repaired that evening, and the pool reopened on Tuesday morning.\n"
                "Which candidate is the most accurate and complete summary?\n"
                "1. The community pool closed on Tuesday because the water was too cold and reopened "
                "Monday before anyone repaired its pump.\n"
                "2. The community pool closed on Monday after its pump broke and reopened Tuesday "
                "after the repair.\n"
                "3. The community pool closed permanently on Monday after its pump broke, despite "
                "repairs being completed later that same evening.\n"
                "Answer with the candidate number only."
            ),
            verifier=Numeric(expected=2),
        ),
        _number(
            id="summarization/selection-clinical-trial",
            prompt=(
                "Source: A 12-week trial enrolled 240 adults, half receiving a new blood-pressure "
                "drug and half receiving standard treatment. Average blood pressure fell by 9 points "
                "with the new drug and by 5 points with standard treatment. Four participants taking "
                "the new drug withdrew because of side effects. The researchers said a larger trial "
                "is needed.\n"
                "Which candidate is the most accurate and complete summary?\n"
                "1. After testing all 240 adults for 12 weeks, researchers proved that the new drug "
                "cured high blood pressure completely, caused no side effects, and required no larger "
                "follow-up study.\n"
                "2. In a definitive 240-person study, standard treatment reduced blood pressure more "
                "than the new drug, nobody withdrew because of side effects, and researchers declared "
                "further trials unnecessary.\n"
                "3. In a 240-person trial, the new drug reduced blood pressure more than standard "
                "treatment, but four participants withdrew and researchers called for a larger study.\n"
                "Answer with the candidate number only."
            ),
            verifier=Numeric(expected=3),
        ),
        _number(
            id="summarization/selection-water-restrictions-fr",
            prompt=(
                "Texte source : Le niveau du réservoir municipal est tombé à 32 % après trois mois "
                "de faibles pluies. À partir du 1er juin, les habitants ne pourront arroser leur jardin "
                "qu'avant 8 heures. Les exploitations agricoles sont exemptées et la mesure sera "
                "réexaminée à la mi-juillet.\n"
                "Quel candidat est le résumé le plus exact et le plus complet ?\n"
                "1. Face à un réservoir tombé à 32 %, la ville limitera dès le 1er juin l'arrosage "
                "des particuliers au matin, avec une exemption agricole et un réexamen mi-juillet.\n"
                "2. La ville interdit immédiatement et jusqu'à la fin de l'année tout arrosage aux "
                "habitants comme aux exploitations agricoles, car le réservoir est tombé à 32 % après "
                "seulement trois jours sans pluie.\n"
                "3. Après la remontée du réservoir à 32 %, la ville supprimera toutes les restrictions "
                "d'arrosage le 1er juin et ne prévoit aucun réexamen de la situation pendant l'été.\n"
                "Réponds uniquement avec le numéro du candidat."
            ),
            verifier=Numeric(expected=1),
        ),
        _boolean(
            id="summarization/faithfulness-wildfire-cause",
            prompt=(
                "Source: Firefighters evacuated 600 homes as a wildfire approached Pine Valley. By "
                "Tuesday evening the fire was 70 percent contained, and no injuries had been reported. "
                "Officials were investigating whether lightning started the fire.\n"
                "Candidate summary: A lightning-caused wildfire forced 600 homes to evacuate near "
                "Pine Valley, but it was 70 percent contained by Tuesday with no reported injuries.\n"
                "Is every claim in the candidate summary supported by the source? Answer only yes or no."
            ),
            verifier=Exact("no"),
        ),
        _boolean(
            id="summarization/faithfulness-acquisition",
            prompt=(
                "Source: Orion Systems agreed to acquire Luma Analytics for 48 million dollars. Luma "
                "will retain its brand and 85 employees. The companies expect the deal to close in the "
                "fourth quarter, subject to regulatory approval.\n"
                "Candidate summary: Orion plans to buy Luma for $48 million while retaining its brand "
                "and staff, but the deal still requires regulatory approval before its expected "
                "fourth-quarter completion.\n"
                "Is every claim in the candidate summary supported by the source? Answer only yes or no."
            ),
            verifier=Exact("yes"),
        ),
        _boolean(
            id="summarization/faithfulness-flight-delay-fr",
            prompt=(
                "Texte source : Le vol transportant 180 passagers a été retardé de trois heures pour "
                "une inspection mécanique. La compagnie a distribué des bons de repas.\n"
                "Résumé candidat : Le vol de 180 passagers a subi un retard de deux heures pour une "
                "inspection, et les voyageurs ont reçu des bons de repas.\n"
                "Toutes les affirmations du résumé candidat sont-elles confirmées par le texte source ? "
                "Réponds uniquement par oui ou non."
            ),
            verifier=Exact("non"),
        ),
        _boolean(
            id="summarization/coverage-bakery-reopening",
            prompt=(
                "Source: The bakery closed on Saturday because its oven broke. The oven was repaired "
                "on Sunday, and the bakery reopened on Monday.\n"
                "Essential facts: why the bakery closed and when it reopened.\n"
                "Candidate summary: The bakery reopened on Monday.\n"
                "Does the candidate summary cover all the essential facts? Answer only yes or no."
            ),
            verifier=Exact("no"),
        ),
        _boolean(
            id="summarization/coverage-chemical-leak",
            prompt=(
                "Source: A chemical leak at the Riverton plant killed fish along two kilometers of the "
                "Grey River. Tests found that the town's drinking water remained safe. The plant stopped "
                "production while environmental authorities investigated the leak.\n"
                "Essential facts: the source of the leak, its environmental impact, the drinking-water "
                "status, and the response.\n"
                "Candidate summary: A leak from the Riverton plant killed fish along part of the Grey "
                "River, although drinking water remained safe; production stopped while authorities "
                "investigated.\n"
                "Does the candidate summary cover all the essential facts? Answer only yes or no."
            ),
            verifier=Exact("yes"),
        ),
        _boolean(
            id="summarization/coverage-prototype-repair",
            prompt=(
                "Source: Luis identified a faulty temperature sensor in Maya's prototype on Wednesday. "
                "Maya then sent the prototype to the repair lab. The lab expects the replacement sensor "
                "on Friday, allowing field tests to resume on Monday.\n"
                "Essential facts: who found the fault, what Maya did next, and the expected repair and "
                "testing timeline.\n"
                "Candidate summary: After Luis found a faulty sensor in Maya's prototype, she sent it "
                "for repair; the replacement is expected Friday and field tests should resume Monday.\n"
                "Does the candidate summary cover all the essential facts? Answer only yes or no."
            ),
            verifier=Exact("yes"),
        ),

        # Classification.
        _chat(
            "classification/customer-praise",
            "Classify the message as exactly one of: complaint, request, praise, cancellation.\n"
            "Message: The replacement arrived today and works perfectly. Thank you for resolving the issue so quickly.\n"
            "Return only the label.",
            _exact("praise"),
        ),
        _number(
            "classification/science-fr",
            "Classe le texte en choisissant la bonne catégorie.\n"
            "Texte : Des chercheurs ont observé une nouvelle exoplanète grâce au télescope spatial.\n"
            "1. politique\n"
            "2. science\n"
            "3. sport\n"
            "4. économie\n"
            "Réponds uniquement avec le numéro de la catégorie.",
            Numeric(expected=2),
        ),
        _number(
            "classification/customer-cancellation",
            "Classify the message by choosing the correct category.\n"
            "Message: Please close my account and end my subscription before the next billing date.\n"
            "1. complaint\n"
            "2. request\n"
            "3. praise\n"
            "4. cancellation\n"
            "Return only the category number.",
            Numeric(expected=4),
        ),
        _chat(
            "classification/economy-fr",
            "Classe le texte dans exactement une catégorie parmi : politique, science, sport, économie.\n"
            "Texte : La banque centrale a relevé ses taux d'intérêt pour ralentir l'inflation.\n"
            "Réponds uniquement avec la catégorie.",
            _exact("économie"),
        ),
        _number(
            "classification/negative-sentiment-number",
            "Classify the sentiment as exactly one of: 1 for \"positive\", 0 for \"neutral\", "
            "-1 for \"negative\".\n"
            "Message: The application crashes every time I try to save a file.\n"
            "Return only the label.",
            Numeric(expected=-1),
        ),
        _number(
            "classification/neutral-sentiment-number",
            "Classify the sentiment as exactly one of: 1 for \"positive\", 0 for \"neutral\", "
            "-1 for \"negative\".\n"
            "Message: The meeting is scheduled for Tuesday at 10 a.m. in room 4.\n"
            "Return only the label.",
            Numeric(expected=0),
        ),
        _number(
            "classification/positive-sentiment-number",
            "Classify the sentiment as exactly one of: 1 for \"positive\", 0 for \"neutral\", "
            "-1 for \"negative\".\n"
            "Message: The new keyboard is comfortable and works exactly as expected.\n"
            "Return only the label.",
            Numeric(expected=1),
        ),
    ),
    observable_sampling=_GREEDY_SAMPLING,
    metrics=(
        MeanScore(sampling=_RANDOM_SAMPLING),
        PassAtK(name="greedy_pass@1", k=1, threshold=1.0, sampling=_GREEDY_SAMPLING),
        PassAtK(k=1, threshold=1.0, sampling=_RANDOM_SAMPLING),
        PassAtK(k=4, threshold=1.0, sampling=_RANDOM_SAMPLING),
        PassAtK(
            name="lpass@4",
            k=4,
            threshold=0.5,
            sampling=_RANDOM_SAMPLING,
        ),
    ),
)
