from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from random import Random
from typing import Any

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_TASK,
    TASK_INFORMATION_EXTRACTION,
)
from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext
from wintermute.tools.params import get_int


_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been before being below
    between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
    him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just me more most mustn't
    my myself no nor not of off on once only or other ought our ours ourselves out over own same she she'd she'll
    she's should shouldn't so some such than that that's the their theirs them themselves then there there's these
    they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're
    we've were weren't what what's when when's where where's which while who who's whom why why's with won't would
    wouldn't may might shall will you you'd you'll you're you've your yours yourself yourselves
    aren couldn didn doesn don hadn hasn haven isn ll m mustn re shouldn ve wasn weren won wouldn
    à ai aient ais ait as au aux avais avait avec avez avions avoir avons aviez avaient ce ces cet cette ceux chaque
    chez comme d dans de des du elle elles en entre es est et étais était étaient été êtes étions étiez être eu eux
    fait font il ils j je l la le les leur leurs lui ma mais me mes moi mon même n ne ni nos notre
    nous on ont ou où par pas pour qu que quel quelle quelles quels qui s sa sans se ses si soi son sont sous sur
    t ta te tes toi ton tous tout toute toutes tu un une vos votre vous y
    """.split()
)

_DIRECTIONS = (
    "key_to_value",
    "value_to_key",
    "position_to_value",
    "value_to_position",
)
_CASE_STYLES = ("lower", "upper", "capitalized")
_LABEL_STYLES = ("number", "letter", "identifier")
_ENTRY_STYLES = (
    "dot",
    "parenthesis",
    "bracket",
    "arrow",
    "round_bracket",
    "equals",
    "colon",
    "double_arrow",
    "csv",
    "tab",
    "json",
    "markdown_row",
)

_ENTRY_FORMATS = {
    "dot": "{label}. {value}",
    "parenthesis": "{label}) {value}",
    "bracket": "[{label}] {value}",
    "arrow": "{label} -> {value}",
    "round_bracket": "({label}) {value}",
    "equals": "{label} = {value}",
    "colon": "{label}: {value}",
    "double_arrow": "{label} => {value}",
    "csv": "{label}, {value}",
    "tab": "{label}\t{value}",
    "markdown_row": "| {label} | {value} |",
}


def _variants(text: str) -> tuple[str, ...]:
    return tuple(text.splitlines())


_INTRODUCTIONS = {
    "en": _variants("""Use the following lookup table.
Read these entries carefully.
Refer to the mapping below.
Inspect the ordered entries.
Consult the table that follows.
Study the list below.
Review the following associations.
Look over these labeled values.
Use the entries shown below.
Examine this lookup list.
Read the mapping presented here.
Consider the following set of entries.
Work from the table below.
Check the labels and values that follow.
Use this ordered collection.
Inspect each entry in the table.
Refer to this list of associations.
Study the labeled items below.
Read through the following mapping.
Consult these ordered values.
Use the reference table below.
Examine the entries in their displayed order.
Look at the following label-value pairs.
Review this small lookup table.
Read the listed entries from top to bottom.
Use the mapping provided below.
Inspect the following reference list.
Consider each labeled item below.
Consult the ordered list that follows.
Study this table before answering.
Use the displayed associations as your reference.
Review the values and their labels below.
Read this lookup table carefully.
Inspect the mapping one entry at a time.
Refer only to the entries shown below.
Use the following list as a reference.
Examine the labels paired with their values.
Study the order and labels of these entries.
Review the table presented underneath.
Read the following reference entries.
Consult this collection of labeled values.
Use the ordered mapping below.
Inspect the list and its labels carefully.
Look through the following reference table.
Study the associations displayed below.
Read each of the following entries.
Use this table to perform the lookup.
Examine the reference mapping below.
Consult the following labeled list.
Review the ordered associations that follow."""),
    "fr": _variants("""Utilise la table de correspondance suivante.
Lis attentivement les entrées suivantes.
Consulte la table ci-dessous.
Examine les entrées ordonnées.
Réfère-toi au tableau suivant.
Étudie la liste ci-dessous.
Observe les associations suivantes.
Parcours ces valeurs étiquetées.
Utilise les entrées affichées ci-dessous.
Examine cette liste de correspondances.
Lis la table présentée ici.
Considère l'ensemble d'entrées suivant.
Travaille à partir du tableau ci-dessous.
Vérifie les labels et les valeurs qui suivent.
Utilise cette collection ordonnée.
Examine chaque entrée du tableau.
Réfère-toi à cette liste d'associations.
Étudie les éléments étiquetés ci-dessous.
Parcours attentivement la table suivante.
Consulte ces valeurs ordonnées.
Utilise le tableau de référence ci-dessous.
Examine les entrées dans l'ordre affiché.
Observe les paires label-valeur suivantes.
Consulte cette petite table de correspondance.
Lis les entrées de haut en bas.
Utilise la correspondance fournie ci-dessous.
Examine la liste de référence suivante.
Considère chaque élément étiqueté ci-dessous.
Consulte la liste ordonnée qui suit.
Étudie ce tableau avant de répondre.
Utilise les associations affichées comme référence.
Observe les valeurs et leurs labels ci-dessous.
Lis attentivement cette table de correspondance.
Examine la table une entrée après l'autre.
Réfère-toi uniquement aux entrées ci-dessous.
Utilise la liste suivante comme référence.
Examine les labels associés à leurs valeurs.
Étudie l'ordre et les labels de ces entrées.
Consulte le tableau présenté ci-après.
Lis les entrées de référence suivantes.
Observe cette collection de valeurs étiquetées.
Utilise la table ordonnée ci-dessous.
Examine soigneusement la liste et ses labels.
Parcours la table de référence suivante.
Étudie les associations affichées ci-dessous.
Lis chacune des entrées suivantes.
Utilise ce tableau pour effectuer la recherche.
Examine la table de référence ci-dessous.
Consulte la liste étiquetée suivante.
Observe les associations ordonnées qui suivent."""),
}

_QUESTIONS = {
    "en": {
        "key_to_value": _variants("""Which value is associated with key '{key}'? Return only the value.
Look up key '{key}'. Answer with its value only.
What value belongs to the label '{key}'? Give only the value.
Find the value for key '{key}'. Return the value alone.
Retrieve the value indexed by '{key}'. Answer only with that value.
For the key '{key}', what is the corresponding value? Give only the value.
Resolve key '{key}' to its value. Output just the value.
Identify the value attached to '{key}'. Return nothing else.
Read the entry labeled '{key}'. Answer with its value only.
Give the value mapped to key '{key}'. Use only the value.
What does key '{key}' map to? Return only the mapped value.
Which item is paired with label '{key}'? Answer with the item only.
Locate key '{key}' and provide its value alone.
Return the value associated with the label '{key}'.
Using the table, find the value for '{key}'. Give only that value.
Key '{key}' points to which value? Answer with the value only.
What value is paired with '{key}'? Return only the value.
Find the content of the entry named '{key}'. Output the content alone.
Determine the value corresponding to key '{key}'. Give just the value.
What is stored under key '{key}'? Return only the stored value.
Read the value beside label '{key}'. Answer with that value alone.
Which value appears next to '{key}'? Give only the value.
Follow the label '{key}' to its value. Return the value only.
Resolve the label '{key}'. Answer with its associated value alone.
State the value corresponding to '{key}' and nothing else.
Extract the value for key '{key}' from the table. Return only it.
What entry belongs to the key '{key}'? Give the entry value only.
Find the table value indexed by '{key}'. Answer with the value alone.
Report the value linked to label '{key}'. Do not include the label.
Search the table for '{key}' and return its value only.
Which word is assigned to key '{key}'? Return only that word.
Identify the item referenced by '{key}'. Answer with the item alone.
Return the lookup result for key '{key}', without any explanation.
What value has been assigned to '{key}'? Give just the value.
Find the exact value matched with label '{key}'. Return only the match.
Give the table entry associated with '{key}', value only.
Read the mapping for key '{key}'. Output only its value.
Which displayed value corresponds to '{key}'? Answer only with the value.
Locate the row labeled '{key}' and return its value alone.
What is the value counterpart of key '{key}'? Give only the counterpart.
Use label '{key}' to retrieve its value. Return only the value.
Determine the mapped word for '{key}'. Answer with that word alone.
Retrieve the table item identified by '{key}'. Output only the item.
State the value found at key '{key}'. Include nothing else.
Which entry follows the label '{key}'? Return only its value.
Find the matching value for key '{key}'. Give only the match.
Provide the value associated with '{key}' and no additional text.
What does the table list for key '{key}'? Answer with the value only.
Read off the value for label '{key}'. Return that value alone.
Complete the lookup for '{key}'. Output only the resulting value."""),
        "value_to_key": _variants("""Which key is associated with value '{value}'? Return only the key.
Find the label for '{value}'. Answer with the label only.
What key points to '{value}'? Give only the key.
Look up the key for value '{value}'. Return the key alone.
Retrieve the label attached to '{value}'. Answer only with the label.
For the value '{value}', what is its key? Give just the key.
Resolve value '{value}' back to its key. Output only the key.
Identify the key paired with '{value}'. Return nothing else.
Find the entry containing '{value}' and answer with its label only.
Give the key mapped to value '{value}'. Use only the key.
Which label leads to '{value}'? Return only that label.
What identifier is paired with the value '{value}'? Answer with the identifier only.
Locate '{value}' and provide its key alone.
Return the key associated with the value '{value}'.
Using the table, find the label for '{value}'. Give only that label.
The value '{value}' belongs to which key? Answer with the key only.
What key is paired with '{value}'? Return only the key.
Find the label of the entry whose value is '{value}'. Output the label alone.
Determine the key corresponding to value '{value}'. Give just the key.
Under which key is '{value}' stored? Return only that key.
Read the label beside value '{value}'. Answer with that label alone.
Which key appears next to '{value}'? Give only the key.
Trace value '{value}' back to its label. Return the label only.
Resolve the value '{value}'. Answer with its associated key alone.
State the key corresponding to '{value}' and nothing else.
Extract the key for value '{value}' from the table. Return only it.
Which label owns the entry '{value}'? Give the label only.
Find the table key whose value is '{value}'. Answer with the key alone.
Report the identifier linked to value '{value}'. Do not include the value.
Search the table for '{value}' and return its key only.
Which label is assigned to the word '{value}'? Return only that label.
Identify the key referencing '{value}'. Answer with the key alone.
Return the reverse lookup result for '{value}', without any explanation.
What key has been assigned to value '{value}'? Give just the key.
Find the exact label matched with '{value}'. Return only the match.
Give the table key associated with '{value}', key only.
Read the reverse mapping for value '{value}'. Output only its key.
Which displayed label corresponds to '{value}'? Answer only with the label.
Locate the row containing '{value}' and return its key alone.
What is the key counterpart of value '{value}'? Give only the counterpart.
Use value '{value}' to retrieve its label. Return only the label.
Determine the identifier mapped to '{value}'. Answer with that identifier alone.
Retrieve the table label for the item '{value}'. Output only the label.
State the key found for value '{value}'. Include nothing else.
Which label precedes the value '{value}'? Return only that label.
Find the matching key for value '{value}'. Give only the match.
Provide the key associated with '{value}' and no additional text.
What label does the table list for '{value}'? Answer with the label only.
Read off the key for value '{value}'. Return that key alone.
Complete the reverse lookup for '{value}'. Output only the resulting key."""),
        "position_to_value": _variants("""Counting from the top starting at 1, what value is at position {position}? Return only the value.
Which value is the {position_ordinal} entry? Answer with the value only.
Read top to bottom. What occupies position {position}? Give only the value.
Find the value in position {position}. Return the value alone.
What value appears at position {position}? Answer only with that value.
Starting from 1 at the top, give the value at position {position}.
Retrieve the {position_ordinal} value in the ordered list. Output only the value.
Identify the item occupying position {position}. Return nothing else.
Read the entry at position {position} and answer with its value only.
Give the value located in slot {position}. Use only the value.
Which value is found at index {position} when counting from 1? Return only it.
What item is the {position_ordinal} one shown? Answer with the item only.
Locate position {position} and provide its value alone.
Return the value stored at ordinal position {position}.
Using the displayed order, find the value at position {position}. Give only that value.
Position {position} contains which value? Answer with the value only.
What value is placed in the {position_ordinal} spot? Return only the value.
Find the content of position {position}. Output the content alone.
Determine the value at list position {position}. Give just the value.
What is written in the {position_ordinal} entry? Return only that value.
Read the value on row {position}, counting from the top. Answer with it alone.
Which value appears in the {position_ordinal} row? Give only the value.
Follow the list to position {position}. Return the value found there.
Resolve ordinal position {position} to its value. Answer with the value alone.
State the value at position {position} and nothing else.
Extract the value from the {position_ordinal} entry. Return only it.
What entry occupies place {position} in the list? Give its value only.
Find the table value at one-based position {position}. Answer with the value alone.
Report the value on line {position}, counting entries from 1.
Search the ordered entries for position {position} and return its value only.
Which word is at the {position_ordinal} position? Return only that word.
Identify the item referenced by position {position}. Answer with the item alone.
Return the positional lookup result for {position}, without any explanation.
What value has been placed at position {position}? Give just the value.
Find the exact value occupying the {position_ordinal} slot. Return only the match.
Give the entry at position {position}, value only.
Read the ordered list at position {position}. Output only its value.
Which displayed value is number {position} from the top? Answer only with the value.
Locate row {position} and return its value alone.
What is the value corresponding to ordinal position {position}? Give only the value.
Use position {position} to retrieve the list value. Return only the value.
Determine the word in the {position_ordinal} entry. Answer with that word alone.
Retrieve the item at one-based position {position}. Output only the item.
State the value found in position {position}. Include nothing else.
Which entry is {position_ordinal} in the displayed order? Return only its value.
Find the matching value for list position {position}. Give only the value.
Provide the value at position {position} and no additional text.
What does the list contain at position {position}? Answer with the value only.
Read off the {position_ordinal} value. Return that value alone.
Complete the positional lookup for {position}. Output only the resulting value."""),
        "value_to_position": _variants("""Counting from the top starting at 1, what is the position of '{value}'? Return only the number.
At which position does '{value}' appear? Answer with the numeric position only.
Read top to bottom. Give the position of '{value}' as a number only.
Find the position of value '{value}'. Return the number alone.
What position contains '{value}'? Answer only with that number.
Starting from 1 at the top, give the position of '{value}'.
Retrieve the one-based position of '{value}'. Output only the number.
Identify the position occupied by '{value}'. Return nothing but the number.
Find the entry containing '{value}' and answer with its position only.
Give the numeric slot where '{value}' is located. Use only the number.
At what one-based index is '{value}' found? Return only the index.
Which numbered position holds '{value}'? Answer with the number only.
Locate '{value}' and provide its position alone.
Return the ordinal position of value '{value}' as a number.
Using the displayed order, find the position of '{value}'. Give only that number.
The value '{value}' occurs in which position? Answer with the number only.
What is the list position of '{value}'? Return only the number.
Find where '{value}' occurs in the ordered entries. Output the position alone.
Determine the one-based position corresponding to '{value}'. Give just the number.
On which numbered row is '{value}' written? Return only the row number.
Read down the list and find '{value}'. Answer with its position alone.
Which position contains the value '{value}'? Give only the number.
Trace '{value}' to its position in the list. Return the position only.
Resolve value '{value}' to its ordinal position. Answer with the number alone.
State the position of '{value}' and nothing else.
Extract the position of value '{value}' from the table. Return only it.
What numbered entry contains '{value}'? Give the entry number only.
Find the one-based table position whose value is '{value}'. Answer with the number alone.
Report the row number linked to value '{value}'. Do not include the value.
Search the ordered entries for '{value}' and return its position only.
Which numbered slot is assigned to the word '{value}'? Return only that number.
Identify the index referencing '{value}'. Answer with the one-based index alone.
Return the reverse positional lookup for '{value}', without any explanation.
What position has been assigned to value '{value}'? Give just the number.
Find the exact row containing '{value}'. Return only its number.
Give the list position associated with '{value}', number only.
Read the order for value '{value}'. Output only its numeric position.
Which displayed position corresponds to '{value}'? Answer only with the number.
Locate the row containing '{value}' and return its position alone.
What is the numeric position counterpart of value '{value}'? Give only the number.
Use value '{value}' to retrieve its position. Return only the number.
Determine the one-based index mapped to '{value}'. Answer with that index alone.
Retrieve the list position for the item '{value}'. Output only the number.
State the position found for value '{value}'. Include nothing else.
Which numbered entry is paired with '{value}'? Return only its number.
Find the matching position for value '{value}'. Give only the number.
Provide the position of '{value}' and no additional text.
What position does the table list for '{value}'? Answer with the number only.
Read off the position of value '{value}'. Return that number alone.
Complete the position lookup for '{value}'. Output only the resulting number."""),
    },
    "fr": {
        "key_to_value": _variants("""Quelle valeur est associée à la clé « {key} » ? Réponds uniquement avec la valeur.
Recherche la clé « {key} ». Donne uniquement sa valeur.
Quelle valeur correspond au label « {key} » ? Donne seulement la valeur.
Trouve la valeur de la clé « {key} ». Réponds avec la valeur seule.
Récupère la valeur indexée par « {key} ». Donne uniquement cette valeur.
Pour la clé « {key} », quelle est la valeur correspondante ? Donne seulement la valeur.
Résous la clé « {key} » vers sa valeur. Écris uniquement la valeur.
Identifie la valeur attachée à « {key} ». Ne renvoie rien d'autre.
Lis l'entrée portant le label « {key} ». Réponds uniquement avec sa valeur.
Donne la valeur associée à la clé « {key} ». Utilise seulement la valeur.
Vers quelle valeur pointe la clé « {key} » ? Réponds uniquement avec cette valeur.
Quel élément est associé au label « {key} » ? Donne uniquement l'élément.
Repère la clé « {key} » et fournis sa valeur seule.
Renvoie la valeur associée au label « {key} ».
À l'aide du tableau, trouve la valeur de « {key} ». Donne uniquement cette valeur.
La clé « {key} » pointe vers quelle valeur ? Réponds seulement avec la valeur.
Quelle valeur est appariée avec « {key} » ? Renvoie uniquement la valeur.
Trouve le contenu de l'entrée nommée « {key} ». Écris seulement ce contenu.
Détermine la valeur correspondant à la clé « {key} ». Donne juste la valeur.
Quelle valeur est stockée sous la clé « {key} » ? Renvoie uniquement cette valeur.
Lis la valeur placée à côté du label « {key} ». Réponds avec cette valeur seule.
Quelle valeur apparaît près de « {key} » ? Donne uniquement la valeur.
Suis le label « {key} » jusqu'à sa valeur. Renvoie seulement la valeur.
Résous le label « {key} ». Réponds uniquement avec la valeur associée.
Indique la valeur correspondant à « {key} » et rien d'autre.
Extrais du tableau la valeur de la clé « {key} ». Renvoie uniquement celle-ci.
Quelle entrée appartient à la clé « {key} » ? Donne seulement sa valeur.
Trouve la valeur du tableau indexée par « {key} ». Réponds avec la valeur seule.
Indique la valeur liée au label « {key} ». N'inclus pas le label.
Cherche « {key} » dans le tableau et renvoie uniquement sa valeur.
Quel mot est attribué à la clé « {key} » ? Renvoie uniquement ce mot.
Identifie l'élément référencé par « {key} ». Réponds avec l'élément seul.
Renvoie le résultat de la recherche pour « {key} », sans explication.
Quelle valeur a été attribuée à « {key} » ? Donne juste la valeur.
Trouve la valeur exacte associée au label « {key} ». Renvoie uniquement la correspondance.
Donne l'entrée du tableau associée à « {key} », valeur uniquement.
Lis la correspondance de la clé « {key} ». Écris uniquement sa valeur.
Quelle valeur affichée correspond à « {key} » ? Réponds seulement avec la valeur.
Repère la ligne portant « {key} » et renvoie uniquement sa valeur.
Quelle est la valeur homologue de la clé « {key} » ? Donne seulement cette valeur.
Utilise le label « {key} » pour retrouver sa valeur. Renvoie uniquement la valeur.
Détermine le mot associé à « {key} ». Réponds avec ce mot seul.
Récupère l'élément du tableau identifié par « {key} ». Écris seulement l'élément.
Indique la valeur trouvée à la clé « {key} ». N'ajoute rien d'autre.
Quelle entrée suit le label « {key} » ? Renvoie uniquement sa valeur.
Trouve la valeur correspondant à la clé « {key} ». Donne uniquement la correspondance.
Fournis la valeur associée à « {key} » sans texte supplémentaire.
Quelle valeur le tableau indique-t-il pour « {key} » ? Réponds uniquement avec la valeur.
Relève la valeur du label « {key} ». Renvoie cette valeur seule.
Effectue la recherche de « {key} ». Écris uniquement la valeur obtenue."""),
        "value_to_key": _variants("""Quelle clé est associée à la valeur « {value} » ? Réponds uniquement avec la clé.
Trouve le label de « {value} ». Donne uniquement le label.
Quelle clé pointe vers « {value} » ? Donne seulement la clé.
Recherche la clé de la valeur « {value} ». Renvoie la clé seule.
Récupère le label attaché à « {value} ». Réponds uniquement avec le label.
Pour la valeur « {value} », quelle est sa clé ? Donne juste la clé.
Remonte de la valeur « {value} » à sa clé. Écris uniquement la clé.
Identifie la clé associée à « {value} ». Ne renvoie rien d'autre.
Trouve l'entrée contenant « {value} » et réponds seulement avec son label.
Donne la clé associée à la valeur « {value} ». Utilise uniquement la clé.
Quel label mène à « {value} » ? Renvoie seulement ce label.
Quel identifiant est associé à la valeur « {value} » ? Réponds uniquement avec l'identifiant.
Repère « {value} » et fournis sa clé seule.
Renvoie la clé associée à la valeur « {value} ».
À l'aide du tableau, trouve le label de « {value} ». Donne uniquement ce label.
La valeur « {value} » appartient à quelle clé ? Réponds seulement avec la clé.
Quelle clé est appariée avec « {value} » ? Renvoie uniquement la clé.
Trouve le label de l'entrée dont la valeur est « {value} ». Écris seulement le label.
Détermine la clé correspondant à la valeur « {value} ». Donne juste la clé.
Sous quelle clé la valeur « {value} » est-elle stockée ? Renvoie uniquement cette clé.
Lis le label placé à côté de « {value} ». Réponds avec ce label seul.
Quelle clé apparaît près de « {value} » ? Donne uniquement la clé.
Remonte de la valeur « {value} » à son label. Renvoie seulement le label.
Résous la valeur « {value} ». Réponds uniquement avec sa clé associée.
Indique la clé correspondant à « {value} » et rien d'autre.
Extrais du tableau la clé de la valeur « {value} ». Renvoie uniquement celle-ci.
Quel label possède l'entrée « {value} » ? Donne seulement le label.
Trouve dans le tableau la clé dont la valeur est « {value} ». Réponds avec la clé seule.
Indique l'identifiant lié à la valeur « {value} ». N'inclus pas la valeur.
Cherche « {value} » dans le tableau et renvoie uniquement sa clé.
Quel label est attribué au mot « {value} » ? Renvoie uniquement ce label.
Identifie la clé qui référence « {value} ». Réponds avec la clé seule.
Renvoie le résultat de la recherche inverse pour « {value} », sans explication.
Quelle clé a été attribuée à la valeur « {value} » ? Donne juste la clé.
Trouve le label exact associé à « {value} ». Renvoie uniquement la correspondance.
Donne la clé du tableau associée à « {value} », clé uniquement.
Lis la correspondance inverse de la valeur « {value} ». Écris uniquement sa clé.
Quel label affiché correspond à « {value} » ? Réponds seulement avec le label.
Repère la ligne contenant « {value} » et renvoie uniquement sa clé.
Quelle est la clé homologue de la valeur « {value} » ? Donne seulement cette clé.
Utilise la valeur « {value} » pour retrouver son label. Renvoie uniquement le label.
Détermine l'identifiant associé à « {value} ». Réponds avec cet identifiant seul.
Récupère le label du tableau pour l'élément « {value} ». Écris seulement le label.
Indique la clé trouvée pour la valeur « {value} ». N'ajoute rien d'autre.
Quel label précède la valeur « {value} » ? Renvoie uniquement ce label.
Trouve la clé correspondant à la valeur « {value} ». Donne uniquement la correspondance.
Fournis la clé associée à « {value} » sans texte supplémentaire.
Quel label le tableau indique-t-il pour « {value} » ? Réponds uniquement avec le label.
Relève la clé de la valeur « {value} ». Renvoie cette clé seule.
Effectue la recherche inverse de « {value} ». Écris uniquement la clé obtenue."""),
        "position_to_value": _variants("""En comptant depuis le haut à partir de 1, quelle valeur est à la position {position} ? Réponds uniquement avec la valeur.
Quelle valeur occupe la position {position} ? Donne uniquement la valeur.
Lis de haut en bas. Que trouve-t-on à la position {position} ? Donne seulement la valeur.
Trouve la valeur en position {position}. Renvoie la valeur seule.
Quelle valeur apparaît à la position {position} ? Réponds uniquement avec cette valeur.
En partant de 1 en haut, donne la valeur à la position {position}.
Récupère la valeur située au rang {position}. Écris uniquement la valeur.
Identifie l'élément qui occupe la position {position}. Ne renvoie rien d'autre.
Lis l'entrée à la position {position} et réponds seulement avec sa valeur.
Donne la valeur située dans l'emplacement {position}. Utilise uniquement la valeur.
Quelle valeur se trouve à l'indice {position} en comptant depuis 1 ? Renvoie seulement celle-ci.
Quel élément est affiché au rang {position} ? Réponds uniquement avec l'élément.
Repère la position {position} et fournis sa valeur seule.
Renvoie la valeur stockée à la position ordinale {position}.
Selon l'ordre affiché, trouve la valeur à la position {position}. Donne uniquement cette valeur.
La position {position} contient quelle valeur ? Réponds seulement avec la valeur.
Quelle valeur est placée au rang {position} ? Renvoie uniquement la valeur.
Trouve le contenu de la position {position}. Écris seulement ce contenu.
Détermine la valeur à la position {position} de la liste. Donne juste la valeur.
Qu'est-il écrit dans l'entrée numéro {position} ? Renvoie uniquement cette valeur.
Lis la valeur de la ligne {position} en comptant depuis le haut. Réponds avec elle seule.
Quelle valeur apparaît sur la ligne {position} ? Donne uniquement la valeur.
Parcours la liste jusqu'à la position {position}. Renvoie la valeur qui s'y trouve.
Résous la position ordinale {position} vers sa valeur. Réponds avec la valeur seule.
Indique la valeur à la position {position} et rien d'autre.
Extrais la valeur de l'entrée numéro {position}. Renvoie uniquement celle-ci.
Quelle entrée occupe la place {position} dans la liste ? Donne seulement sa valeur.
Trouve la valeur du tableau à la position {position}, en partant de 1. Réponds avec la valeur seule.
Indique la valeur de la ligne {position} en numérotant depuis 1.
Cherche la position {position} dans les entrées ordonnées et renvoie uniquement sa valeur.
Quel mot est à la position {position} ? Renvoie uniquement ce mot.
Identifie l'élément référencé par la position {position}. Réponds avec l'élément seul.
Renvoie le résultat de la recherche positionnelle pour {position}, sans explication.
Quelle valeur a été placée à la position {position} ? Donne juste la valeur.
Trouve la valeur exacte qui occupe l'emplacement {position}. Renvoie uniquement la correspondance.
Donne l'entrée à la position {position}, valeur uniquement.
Lis la liste ordonnée à la position {position}. Écris uniquement sa valeur.
Quelle valeur affichée est la numéro {position} depuis le haut ? Réponds seulement avec la valeur.
Repère la ligne {position} et renvoie uniquement sa valeur.
Quelle valeur correspond à la position ordinale {position} ? Donne seulement la valeur.
Utilise la position {position} pour retrouver la valeur de la liste. Renvoie uniquement la valeur.
Détermine le mot de l'entrée numéro {position}. Réponds avec ce mot seul.
Récupère l'élément à la position {position} en comptant depuis 1. Écris seulement l'élément.
Indique la valeur trouvée à la position {position}. N'ajoute rien d'autre.
Quelle entrée est numéro {position} dans l'ordre affiché ? Renvoie uniquement sa valeur.
Trouve la valeur correspondant à la position {position} de la liste. Donne uniquement la valeur.
Fournis la valeur à la position {position} sans texte supplémentaire.
Que contient la liste à la position {position} ? Réponds uniquement avec la valeur.
Relève la valeur en position {position}. Renvoie cette valeur seule.
Effectue la recherche positionnelle pour {position}. Écris uniquement la valeur obtenue."""),
        "value_to_position": _variants("""En comptant depuis le haut à partir de 1, quelle est la position de « {value} » ? Réponds uniquement avec le nombre.
À quelle position apparaît « {value} » ? Donne uniquement la position numérique.
Lis de haut en bas. Donne la position de « {value} » sous forme de nombre uniquement.
Trouve la position de la valeur « {value} ». Renvoie le nombre seul.
Quelle position contient « {value} » ? Réponds uniquement avec ce nombre.
En partant de 1 en haut, donne la position de « {value} ».
Récupère la position de « {value} » en comptant depuis 1. Écris seulement le nombre.
Identifie la position occupée par « {value} ». Ne renvoie rien d'autre que le nombre.
Trouve l'entrée contenant « {value} » et réponds uniquement avec sa position.
Donne l'emplacement numérique où se trouve « {value} ». Utilise seulement le nombre.
À quel indice, en partant de 1, trouve-t-on « {value} » ? Renvoie uniquement l'indice.
Quelle position numérotée contient « {value} » ? Réponds seulement avec le nombre.
Repère « {value} » et fournis sa position seule.
Renvoie sous forme numérique la position ordinale de « {value} ».
Selon l'ordre affiché, trouve la position de « {value} ». Donne uniquement ce nombre.
La valeur « {value} » apparaît à quelle position ? Réponds seulement avec le nombre.
Quelle est la position de « {value} » dans la liste ? Renvoie uniquement le nombre.
Trouve où apparaît « {value} » dans les entrées ordonnées. Écris seulement la position.
Détermine la position de « {value} » en comptant depuis 1. Donne juste le nombre.
Sur quelle ligne numérotée « {value} » est-il écrit ? Renvoie uniquement le numéro.
Parcours la liste et trouve « {value} ». Réponds avec sa position seule.
Quelle position contient la valeur « {value} » ? Donne uniquement le nombre.
Retrouve la position de « {value} » dans la liste. Renvoie seulement la position.
Résous la valeur « {value} » vers sa position ordinale. Réponds avec le nombre seul.
Indique la position de « {value} » et rien d'autre.
Extrais du tableau la position de la valeur « {value} ». Renvoie uniquement celle-ci.
Quelle entrée numérotée contient « {value} » ? Donne seulement son numéro.
Trouve la position du tableau dont la valeur est « {value} », en partant de 1. Réponds avec le nombre seul.
Indique le numéro de ligne lié à la valeur « {value} ». N'inclus pas la valeur.
Cherche « {value} » dans les entrées ordonnées et renvoie uniquement sa position.
Quel emplacement numéroté est attribué au mot « {value} » ? Renvoie seulement ce nombre.
Identifie l'indice qui référence « {value} ». Réponds avec l'indice seul, en partant de 1.
Renvoie la recherche positionnelle inverse de « {value} », sans explication.
Quelle position a été attribuée à la valeur « {value} » ? Donne juste le nombre.
Trouve la ligne exacte contenant « {value} ». Renvoie uniquement son numéro.
Donne la position de liste associée à « {value} », nombre uniquement.
Lis l'ordre pour la valeur « {value} ». Écris uniquement sa position numérique.
Quelle position affichée correspond à « {value} » ? Réponds seulement avec le nombre.
Repère la ligne contenant « {value} » et renvoie uniquement sa position.
Quel est l'emplacement numérique correspondant à « {value} » ? Donne seulement le nombre.
Utilise la valeur « {value} » pour retrouver sa position. Renvoie uniquement le nombre.
Détermine l'indice de « {value} » en comptant depuis 1. Réponds avec cet indice seul.
Récupère la position dans la liste de l'élément « {value} ». Écris seulement le nombre.
Indique la position trouvée pour la valeur « {value} ». N'ajoute rien d'autre.
Quelle entrée numérotée est associée à « {value} » ? Renvoie uniquement son numéro.
Trouve la position correspondant à la valeur « {value} ». Donne uniquement le nombre.
Fournis la position de « {value} » sans texte supplémentaire.
Quelle position le tableau indique-t-il pour « {value} » ? Réponds uniquement avec le nombre.
Relève la position de la valeur « {value} ». Renvoie ce nombre seul.
Effectue la recherche de position pour « {value} ». Écris uniquement le nombre obtenu."""),
    },
}


class LookupRetrievalRecordTransform(RecordTransform):
    """Create a synthetic table-lookup SFT example from words in a record.

    It samples distinct words, assigns labels, and asks for a value, key, or
    position in English or French.  The result is a single information-
    extraction prompt/completion pair, not a projection of the source fields.
    """

    KIND = "lookup_retrieval"

    def __init__(self, *, min_items: int = 4, max_items: int = 10) -> None:
        if min_items < 4 or max_items < min_items:
            raise ValueError("lookup_retrieval requires 4 <= min_items <= max_items")
        self.min_items = min_items
        self.max_items = max_items

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> LookupRetrievalRecordTransform:
        unexpected = sorted(set(params).difference({"min_items", "max_items"}))
        if unexpected:
            raise ValueError(f"lookup_retrieval does not support parameters: {', '.join(unexpected)}")
        return cls(
            min_items=get_int(params, "min_items", 4),
            max_items=get_int(params, "max_items", 10),
        )

    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> Iterable[DataRecord]:
        words = self._record_words(record)
        if len(words) < self.min_items:
            return

        rng = context.rng
        item_count = rng.randint(self.min_items, min(self.max_items, len(words)))
        case_style = rng.choice(_CASE_STYLES)
        values = [
            self._apply_case(word, case_style)
            for word in rng.sample(words, item_count)
        ]
        direction = rng.choice(_DIRECTIONS)
        labels = self._labels(
            rng.choice(_LABEL_STYLES),
            item_count,
            positional=direction in {"position_to_value", "value_to_position"},
            rng=rng,
        )
        target_index = rng.randrange(item_count)
        language = rng.choice(("en", "fr"))

        prompt = self._prompt(
            labels=labels,
            values=values,
            direction=direction,
            target_index=target_index,
            language=language,
            rng=rng,
        )
        if direction in {"key_to_value", "position_to_value"}:
            completion = values[target_index]
        elif direction == "value_to_key":
            completion = labels[target_index]
        else:
            completion = str(target_index + 1)

        yield DataRecord(
            record_id=f"{record.record_id}.lookup-{direction}",
            fields={
                FLD_GENERIC_PROMPT: prompt,
                FLD_GENERIC_COMPLETION: completion,
                FLD_GENERIC_TASK: TASK_INFORMATION_EXTRACTION,
            },
        )

    def _record_words(self, record: DataRecord) -> list[str]:
        words: list[str] = []
        seen: set[str] = set()
        for field_name in sorted(record.fields):
            for word in _WORD_PATTERN.findall(record.fields[field_name]):
                normalized = word.casefold()
                if len(normalized) < 2 or normalized in _STOPWORDS or normalized in seen:
                    continue
                seen.add(normalized)
                words.append(word)
        return words

    def _labels(
        self,
        style: str,
        count: int,
        *,
        positional: bool,
        rng: Random,
    ) -> list[str]:
        if style == "number":
            if positional:
                return [str(index) for index in range(1, count + 1)]
            return [str(value) for value in rng.sample(range(11, 1000), count)]

        if style == "letter":
            candidates = [self._letter_label(index) for index in range(max(26, count * 3))]
            return candidates[:count] if positional else rng.sample(candidates, count)

        identifiers: list[str] = []
        while len(identifiers) < count:
            candidate = f"{rng.choice('BCDFGHJKLMNPQRSTVWXYZ')}{rng.randrange(10, 100)}"
            if candidate not in identifiers:
                identifiers.append(candidate)
        return identifiers

    def _apply_case(self, value: str, style: str) -> str:
        if style == "lower":
            return value.lower()
        if style == "upper":
            return value.upper()
        return value.capitalize()

    def _letter_label(self, index: int) -> str:
        label = ""
        while True:
            index, remainder = divmod(index, 26)
            label = chr(ord("A") + remainder) + label
            if index == 0:
                return label
            index -= 1

    def _prompt(
        self,
        *,
        labels: list[str],
        values: list[str],
        direction: str,
        target_index: int,
        language: str,
        rng: Random,
    ) -> str:
        entries = self._render_entries(labels, values, rng.choice(_ENTRY_STYLES))
        question = rng.choice(_QUESTIONS[language][direction]).format(
            key=labels[target_index],
            value=values[target_index],
            position=target_index + 1,
            position_ordinal=self._english_ordinal(target_index + 1),
        )
        return f"{rng.choice(_INTRODUCTIONS[language])}\n{entries}\n{question}"

    def _render_entries(self, labels: list[str], values: list[str], style: str) -> str:
        if style == "json":
            return "\n".join(
                json.dumps({"key": label, "value": value}, ensure_ascii=False)
                for label, value in zip(labels, values)
            )

        entry_format = _ENTRY_FORMATS[style]
        return "\n".join(
            entry_format.format(label=label, value=value)
            for label, value in zip(labels, values)
        )

    def _english_ordinal(self, value: int) -> str:
        if 10 < value % 100 < 14:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"
