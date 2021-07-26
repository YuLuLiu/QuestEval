# Usage

### Code Example
```
from questeval.mask_questeval import Mask_QuestEval
questeval = Mask_QuestEval()

source_1 = "The 2020 Summer Olympics, officially the Games of the XXXII Olympiad and branded as Tokyo 2020, is an ongoing international multi-sport event that is currently being held from 23 July to 8 August 2021 in Tokyo, Japan, with some preliminary events beginning on 21 July."
prediction_1 = "The 2020 Summer Olympics happens in Tokyo, Japan."

score = questeval.corpus_questeval(hypothesis=[prediction_1], sources=[source_1])
print(score)

log_src = questeval.open_log_from_text(source_1)
log_pred = questeval.open_log_from_text(prediction_1)
```
### Expected Output
```
#TODO: copy expected output here
```
