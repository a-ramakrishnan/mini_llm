# Dataset recipes: which datasets each training stage reads.
#
# A spec is "name[@subset][%rows][#column]": a Hugging Face dataset, one of its subsets, only its first `rows`
# training rows (streamed, so the rest isn't downloaded), and the column that holds the text (default "text").

FINEWEB = "BaoMinhTran/mini-fineweb-edu"       # educational web pages; subsets debug < tiny < small < base < full
COSMOPEDIA = "HuggingFaceTB/cosmopedia"        # synthetic textbooks, stories and how-to articles

# the five Cosmopedia subsets, the same in stage 1 and stage 2
_COSMOPEDIA = [f"{COSMOPEDIA}@stories%300000", f"{COSMOPEDIA}@wikihow", f"{COSMOPEDIA}@openstax",
               f"{COSMOPEDIA}@stanford%100000", f"{COSMOPEDIA}@khanacademy"]

RECIPES = {
    # the mini path: ~129M tokens of web text, enough for a small model on a laptop
    "mini": {"specs": [f"{FINEWEB}@tiny"],
             "about": "Mini FineWeb-Edu tiny: the mini path's pretraining data"},
    # more of the same, for Day 4's "continue on more data" on the mini path
    "mini-more": {"specs": [f"{FINEWEB}@tiny", f"{COSMOPEDIA}@khanacademy"], "held_out_from": "mini",
                  "about": "the mini recipe plus Khan Academy-style lessons"},
    # stage 1 as it was run: 500M tokens seen
    "stage1": {"specs": [f"{FINEWEB}@small"] + _COSMOPEDIA,
               "about": "Mini FineWeb-Edu small + five Cosmopedia subsets"},
    # stage 2 as it was run: 2.5B tokens seen; stage 1's validation documents are removed from the full subset
    "stage2": {"specs": [f"{FINEWEB}@full"] + _COSMOPEDIA, "held_out_from": "stage1",
               "about": "Mini FineWeb-Edu full + the same Cosmopedia subsets"},
}
