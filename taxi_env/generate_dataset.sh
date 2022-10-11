# Generate taxi env datasets

NUM_TRAIN=100000 #25000
NUM_TEST=10000   #5000
NUM_VAL=10000    #5000

python -m taxi_env.generator \
  --dir_name='taxi_dataset' \
  --seed=17 \
  --num_train=$NUM_TRAIN \
  --num_test=$NUM_TEST \
  --num_val=$NUM_VAL \
  --max_program_length=50 \
  --max_program_depth=6 \
  --min_max_demo_length_for_program=2 \
  --min_demo_length=8 \
  --max_demo_length=20 \
  --num_demo_per_program=10 \
  --num_test_demo_per_program=5 \
  --max_demo_generation_trial=100
