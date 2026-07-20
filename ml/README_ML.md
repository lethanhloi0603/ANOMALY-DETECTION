# ML worker commands

```bash
python cert_prepare.py --cert-dir ../data/cert4.2 --out-dir ../data/processed --holdout-users 2 --min-logs 50
python train.py --scope global --events-csv ../data/processed/global_events.csv --model-dir ../models/global --window-size 30 --epochs 3
python train.py --scope personalized --db ../data/anomaly.db --user-id USER_ID --model-dir ../models/personalized/USER_ID --window-size 30 --epochs 3 --min-events 50
python predict.py --db ../data/anomaly.db --user-id USER_ID --scope global --model-dir ../models/global --window-size 30
```
