To run:

- Put checkpoints_mlt in /web
- Put recommendation.pkl and rf_200.pkl in db/
- Copy nets and utils into /web
- Make sure utils/bbox/make.sh is run and that bbox.so and nms.so exists in
  utils/bbox
- Make sure the latest pickles are in web/pickles

Finally run python main.py
