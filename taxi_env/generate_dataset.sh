# Generate taxi env datasets
python -m taxi_env.generator \
  --dir_name='taxi_dataset' \
  --seed=123 \
  --num_train=25000 \
  --num_test=5000 \
  --num_val=5000 \
  --max_program_length=50 \
  --max_program_depth=6 \
  --min_max_demo_length_for_program=2 \
  --min_demo_length=8 \
  --max_demo_length=20 \
  --num_demo_per_program=10 \
  --num_test_demo_per_program=5 \
  --max_demo_generation_trial=100
