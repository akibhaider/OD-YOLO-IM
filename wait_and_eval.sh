#!/bin/bash
# Wait for stratified_resplit.py to finish, then run ensemble eval
echo "Watching for training completion..."
while pgrep -f "stratified_resplit.py" > /dev/null 2>&1; do
    EPOCH=$(grep -c "^                   all" stratified_train.log 2>/dev/null || echo 0)
    LAST=$(grep "^                   all" stratified_train.log 2>/dev/null | tail -1)
    echo "[$(date +%H:%M:%S)] Training running... epoch $EPOCH done. Last: $LAST"
    sleep 60
done
echo ""
echo "====================================="
echo "Training complete! Starting ensemble evaluation..."
echo "====================================="
sleep 5

CUDA_VISIBLE_DEVICES='1' python3 /mnt/Data/AKIB/YOLO-OD-IM/stratified_ensemble_eval.py
