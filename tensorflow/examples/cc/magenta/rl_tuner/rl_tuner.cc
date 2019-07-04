/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/examples/cc/magenta/rl_tuner/rl_tuner.h"

#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_set>

#include "tensorflow/examples/cc/magenta/rl_tuner/const.h"

#define RANDOM_ACTION_PROBABILITY 0.1
#define STORE_EVERY_NTH 1
#define TRAIN_EVERY_NTH 5
#define DISCOUNT_RATE 0.5
#define TARGET_NETWORK_UPDATE_RATE 0.01
#define EXPLORATION_PERIOD 5000

#define REWARD_SCALER 0.1
#define C_MAJOR_KEY {0, 1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 25, 26, 28, 30, 31, 33, 35, 37}

#define GRAPH_PATH "/tmp/magenta_frozen.pb"

using tensorflow::DT_FLOAT;
using tensorflow::DT_UINT8;
using tensorflow::Output;
using tensorflow::TensorShape;
using std::vector;
using namespace tensorflow::ops;
using namespace tensorflow::ops::internal;

namespace tensorflow {

RLTuner::RLTuner()
    : scope(Scope::NewRootScope()), session(scope),
      q_network(NoteRNN(scope, session)),
      target_q_network(NoteRNN(scope, session)),
      reward_rnn(NoteRNN(scope, session)) {
  this->Init();
}

RLTuner::~RLTuner() {
}

Status RLTuner::Init() {
  this->num_actions = INPUT_SIZE;
  this->actions_executed_so_far = 0;
  this->num_steps = TRAINING_STEPS;
  this->num_times_train_called = 0;
  this->num_times_store_called = 0;

  this->note_rnn_reward_last_n = 0.0;
  this->music_theory_reward_last_n = 0.0;

  this->BuildGraph();

  this->q_network.Init();
  this->q_network.Restore(GRAPH_PATH);
  this->target_q_network.Init();
  this->target_q_network.Restore(GRAPH_PATH);
  this->reward_rnn.Init();
  this->reward_rnn.Restore(GRAPH_PATH);

  return Status::OK();
}

Status RLTuner::BuildGraph() {
  // reward_scores
  this->reward_scores = Identity(this->scope, this->reward_rnn);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // action_scores
  this->action_scores = Identity(this->scope, this->q_network);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // taking_action
  this->action_softmax = Softmax(this->scope, this->action_scores);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->predicted_actions = OneHot(this->scope,
                                   ArgMax(this->scope, this->action_scores, 1),
                                   this->num_actions, 1.0f, 0.0f);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // estimating_future_rewards
  this->next_action_scores = StopGradient(this->scope, this->target_q_network);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // Rewards are observed from the environment and are fed in later.
  this->rewards = Placeholder(this->scope, DT_FLOAT,
                              Placeholder::Shape({MINIBATCH_SIZE}));
  LOG(INFO) << "Node building status: " << this->scope.status();

  // target_vals
  this->target_vals = ReduceMax(this->scope, this->next_action_scores, {1});
  LOG(INFO) << "Node building status: " << this->scope.status();

  // Total rewards are the observed rewards plus
  // discounted estimated future rewards.
  this->future_rewards = Add(this->scope, this->rewards,
                             Mul(this->scope, this->target_vals,
                                 Cast(this->scope, DISCOUNT_RATE, DT_FLOAT)));
  LOG(INFO) << "Node building status: " << this->scope.status();

  // q_value_prediction
  this->action_mask = Placeholder(this->scope, DT_FLOAT,
                                  Placeholder::Shape({INPUT_SIZE, MINIBATCH_SIZE}));
  LOG(INFO) << "Node building status: " << this->scope.status();

  this->masked_action_scores = ReduceSum(this->scope,
                                         MatMul(this->scope, this->action_scores, this->action_mask),
                                         {1});
  LOG(INFO) << "Node building status: " << this->scope.status();

  this->temp_diff = Sub(this->scope, this->masked_action_scores, this->future_rewards);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // Prediction error is the mean squared error between the reward the
  // network actually received for a given action, and what it expected to
  // receive.
  this->prediction_error = ReduceMean(this->scope, Square(this->scope, this->temp_diff), {0});
  LOG(INFO) << "Node building status: " << this->scope.status();

  // Gradients
  TF_CHECK_OK(AddSymbolicGradients(this->scope, {this->prediction_error},
                                   {this->q_network.w, this->q_network.b, this->q_network.w_y, this->q_network.b_y},
                                   &this->grad_outputs));
  LOG(INFO) << "Node building status: " << this->scope.status();

  this->lr = Cast(this->scope, 0.03, DT_FLOAT);

  // Clip gradients. TODO(Rock)
      // for i, (grad, var) in enumerate(this->gradients):
      //   if grad is not None:
      //     this->gradients[i] = (tf.clip_by_norm(grad, 5), var)

  // alternative of ApplyAdagrad
  this->apply_w = ApplyAdagradTrick(this->scope, this->q_network.w,
                                    this->q_network.ada_w, this->lr, this->grad_outputs[0]);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->apply_b = ApplyAdagradTrick(this->scope, this->q_network.b, this->q_network.ada_b,
                                    this->lr, this->grad_outputs[1]);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->apply_w_y = ApplyAdagradTrick(this->scope, this->q_network.w_y, this->q_network.ada_w_y,
                                      this->lr, this->grad_outputs[2]);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->apply_b_y = ApplyAdagradTrick(this->scope, this->q_network.b_y, this->q_network.ada_b_y,
                                      this->lr, this->grad_outputs[3]);
  LOG(INFO) << "Node building status: " << this->scope.status();

  // target_network_update: w, b, w_y, b_y
  this->update_target_w = this->AssignSub(this->target_q_network.w, this->q_network.w);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->update_target_b = this->AssignSub(this->target_q_network.b, this->q_network.b);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->update_target_w_y = this->AssignSub(this->target_q_network.w_y, this->q_network.w_y);
  LOG(INFO) << "Node building status: " << this->scope.status();
  this->update_target_b_y = this->AssignSub(this->target_q_network.b_y, this->q_network.b_y);
  LOG(INFO) << "Node building status: " << this->scope.status();

  return this->scope.status();
}

// target = (1 - TARGET_NETWORK_UPDATE_RATE) * target + TARGET_NETWORK_UPDATE_RATE * source;
Output RLTuner::AssignSub(Output &target, const Output &source) {
  target = Add(this->scope,
                  Mul(this->scope, Cast(this->scope, 1 - TARGET_NETWORK_UPDATE_RATE,  DT_FLOAT), target),
                  Mul(this->scope, Cast(this->scope, TARGET_NETWORK_UPDATE_RATE,  DT_FLOAT), source));

  return target;
}

void RLTuner::Train() {
  // last_observation
  this->last_observation = this->PrimeInternalModels();

  // loop
  Tensor action, new_observation, reward_scores;
  for (int step = 0; step < this->num_steps; step++) {
    Tensor state_h = this->q_network.h_prev_tensor;
    Tensor state_c = this->q_network.cs_prev_tensor;

    std::tie(action, new_observation, reward_scores) = this->Action();

    Tensor new_state_h = this->q_network.h_prev_tensor;
    Tensor new_state_c = this->q_network.cs_prev_tensor;

    Tensor new_reward_state_h = this->reward_rnn.h_prev_tensor;
    Tensor new_reward_state_c = this->reward_rnn.cs_prev_tensor;

    // reward
    Tensor reward = this->CollectReward(last_observation, new_observation, reward_scores);

    this->Store(last_observation, state_h, state_c, action, reward, new_observation,
                 new_state_h, new_state_c, new_reward_state_h, new_reward_state_c);

    this->TrainingStep();

    if (step % 1000 == 0) {
      LOG(INFO) << "Print step: " << step << ", reward_scores: " << reward_scores.DebugString();
      LOG(INFO) << "Print step: " << step << ", new_observation: " << new_observation.DebugString();
    }

    // Update current state as last state. TODO
    this->last_observation = new_observation;
  }
}

// return a random value between [0, 1)
double RLTuner::Random() {
  std::mt19937_64 rng;

  // initialize the random number generator with time-dependent seed
  uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
  rng.seed(ss);

  // initialize a uniform distribution between 0 and 1
  std::uniform_real_distribution<double> unif(0, 1);

  // ready to generate random numbers
  return unif (rng);
}

std::tuple<Tensor, Tensor, Tensor> RLTuner::Action() {
  this->actions_executed_so_far += 1;

  this->exploration_p = this->LinearAnnealing(this->actions_executed_so_far,
                                            EXPLORATION_PERIOD,
                                            1.0, RANDOM_ACTION_PROBABILITY);

  // Run
  vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({
                           {this->q_network.x, this->last_observation},
                           {this->q_network.y, this->q_network.y_tensor},
                           {this->q_network.h_prev, this->q_network.h_prev_tensor},
                           {this->q_network.cs_prev, this->q_network.cs_prev_tensor},
                           {this->reward_rnn.x, this->last_observation},
                           {this->reward_rnn.y, this->reward_rnn.y_tensor},
                           {this->reward_rnn.h_prev, this->reward_rnn.h_prev_tensor},
                           {this->reward_rnn.cs_prev, this->reward_rnn.cs_prev_tensor},
                          },
                          {
                            this->predicted_actions, this->action_softmax, this->reward_scores,
                            this->q_network.block_lstm->h, this->q_network.block_lstm->cs,
                            this->reward_rnn.block_lstm->h, this->reward_rnn.block_lstm->cs
                          },
                          {},
                          &outputs));

  // update h_prev, cs_prev
  this->q_network.UpdateState(outputs[3], outputs[4]);
  this->reward_rnn.UpdateState(outputs[5], outputs[6]);

  // return
  if (this->Random() < exploration_p) {
    Tensor note = this->GetRandomNote();
    return std::make_tuple(note, note, outputs[2]);
  } else {
    // Reshape and return
    Tensor action;
    CHECK(action.CopyFrom(outputs[0], TensorShape({1, MINIBATCH_SIZE, INPUT_SIZE})));
    return std::make_tuple(action, action, outputs[2]);
  }
}

void RLTuner::TrainingStep() {
  if (this->num_times_train_called % TRAIN_EVERY_NTH == 0) {
    if (this->experience.size() < MINIBATCH_SIZE)
      return;

    // radom samples, TODO
    // std::vector<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> samples;
    // std::sample(this->experience.begin(), this->experience.end(), std::back_inserter(samples),
    //             MINIBATCH_SIZE, std::mt19937{std::random_device{}()});

    // for (auto it = samples.cbegin(); it != samples.cend(); it++) {
    //   // TIME_LEN = 1 and BATCH_SIZE = 1
    //   // last_observation and new_observation are in the shape of {TIME_LEN, BATCH_SIZE, INPUT_SIZE}
    //   // *state_* are in the shape of {BATCH_SIZE, NUM_UNIT}
    //   // action is in the shape of {BATCH_SIZE, INPUT_SIZE}
    //   // reward is in the shape of {BATCH_SIZE, NUM_UNIT}
    //   Tensor last_observation, state_h, state_c, action, reward, new_observation,
    //              new_state_h, new_state_c, new_reward_state_h, new_reward_state_c;
    //   std::tie(last_observation, state_h, state_c, action, reward, new_observation,
    //              new_state_h, new_state_c, new_reward_state_h, new_reward_state_c) = *it;

    //   // Fill in tensors


    srand(time(NULL));
    int rand_index = rand() % this->experience.size();
    Tensor observations, state_h, state_c, action, rewards, new_observations,
                 new_state_h, new_state_c, new_reward_state_h, new_reward_state_c;
    std::tie(observations, state_h, state_c, action, rewards, new_observations,
                 new_state_h, new_state_c, new_reward_state_h, new_reward_state_c) = this->experience[rand_index];

    Tensor action_mask;
    CHECK(action_mask.CopyFrom(action, TensorShape({INPUT_SIZE, MINIBATCH_SIZE})));

    // Backprop
    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({
                              {this->q_network.x, observations},
                              {this->q_network.y, this->q_network.y_tensor},
                              {this->q_network.h_prev, state_h},
                              {this->q_network.cs_prev, state_c},

                              {this->target_q_network.x, new_observations},
                              {this->target_q_network.y, this->target_q_network.y_tensor},
                              {this->target_q_network.h_prev, new_state_h},
                              {this->target_q_network.cs_prev, new_state_c},

                              {this->action_mask, action_mask},
                              {this->rewards, rewards}
                            },
                            {
                              this->prediction_error,
                              this->apply_w, this->apply_b, this->apply_w_y, this->apply_b_y,
                              this->future_rewards
                            },
                            {},
                            &outputs));

    // //Tensors for placeholders
    // Tensor observations(DT_FLOAT, TensorShape({1, MINIBATCH_SIZE, INPUT_SIZE}));
    // Tensor new_observations(DT_FLOAT, TensorShape({1, MINIBATCH_SIZE, INPUT_SIZE}));
    // Tensor action_mask(DT_FLOAT, TensorShape({INPUT_SIZE, MINIBATCH_SIZE}));
    // Tensor rewards(DT_FLOAT, TensorShape({MINIBATCH_SIZE}));

    // TF_CHECK_OK(session.Run({
    //                           {this->q_network.x, observations},
    //                           {this->q_network.y, this->q_network.y_tensor},
    //                           {this->q_network.h_prev, this->q_network.h_prev_tensor},
    //                           {this->q_network.cs_prev, this->q_network.cs_prev_tensor},

    //                           {this->target_q_network.x, new_observations},
    //                           {this->target_q_network.y, this->target_q_network.y_tensor},
    //                           {this->target_q_network.h_prev, this->target_q_network.h_prev_tensor},
    //                           {this->target_q_network.cs_prev, this->target_q_network.cs_prev_tensor},

    //                           {this->action_mask, action_mask},
    //                           {this->rewards, rewards}
    //                         },
    //                         {
    //                           this->prediction_error,
    //                           this->apply_w, this->apply_b, this->apply_w_y, this->apply_b_y,
    //                           this->future_rewards
    //                         },
    //                         {},
    //                         &outputs));
    if (this->num_times_train_called % 1000 == 0) {
      LOG(INFO) << "Print num_times_train_called: " << this->num_times_train_called
                << ", prediction_error: " << outputs[0].DebugString();
      LOG(INFO) << "Print num_times_train_called 111: " << this->num_times_train_called
                << ", future_rewards: " << outputs[5].DebugString();
    }

    // target_network_update
    TF_CHECK_OK(session.Run({},
                            {this->update_target_w, this->update_target_b,
                             this->update_target_w_y, this->update_target_b_y},
                            {},
                            &outputs));
  }

  this->num_times_train_called++;
}

// Get one random note in the shape of {1, 1, INPUT_SIZE}
// with the format of one-hot
Tensor RLTuner::GetRandomNote() {
  srand(time(NULL));

  int randValue = rand() % INPUT_SIZE;

  Tensor ret_tensor(DT_FLOAT, TensorShape({1, MINIBATCH_SIZE, INPUT_SIZE}));

  // Assign a 1 x INPUT_SIZE * 1 matrix (really vector) to a slice of size
  Eigen::Tensor<float, 2, Eigen::RowMajor> m(1, INPUT_SIZE);
  m.setZero();

  // one-hot processing for one character
  m(0, randValue) = 1.0f;

  auto e_2d = ret_tensor.shaped<float, 2>({1, INPUT_SIZE});

  // set e_2d
  Eigen::DSizes<Eigen::DenseIndex, 2> indices(0, 0);
  Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, INPUT_SIZE);
  e_2d.slice(indices, sizes) = m;

  return ret_tensor;
}

Tensor RLTuner::PrimeInternalModels() {
  // for priming_mode == 'random_note'
  this->PrimeInternalModel(this->target_q_network);
  this->PrimeInternalModel(this->reward_rnn);
  return this->PrimeInternalModel(this->q_network);
}

Tensor RLTuner::PrimeInternalModel(const NoteRNN &q_network) {
  // zero out variables

  // return
  return this->GetRandomNote();
}

double RLTuner::LinearAnnealing(int n, int total, double p_initial, double p_final) {
  if (n >= total)
    return p_final;
  else
    return p_initial - (static_cast<double>(n) * (p_initial - p_final)) / (static_cast<double>(total));
}

void RLTuner::Store(const Tensor &observation, const Tensor &state_h, const Tensor &state_c,
             const Tensor &action, const Tensor &reward,
             const Tensor &newobservation, const Tensor &newstate_h, const Tensor &newstate_c,
             const Tensor &new_reward_state_h, const Tensor &new_reward_state_c) {
    if (this->num_times_store_called % STORE_EVERY_NTH == 0) {
      this->experience.emplace_back(observation, state_h, state_c, action, reward,
                              newobservation, newstate_h, newstate_c, new_reward_state_h, new_reward_state_c);
    }

    this->num_times_store_called += 1;
}

// RewardFromRewardRnnScores
double RLTuner::RewardFromRewardRnnScores(const Tensor &action, const Tensor &reward_scores) {
#if TESTING
  // Ref: https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/test/cxx11_tensor_argmax.cpp

  // Test code
  Eigen::Tensor<float, 4, Eigen::RowMajor> tensor(2, 3, 5, 7);
  tensor.setRandom();
  tensor = (tensor + tensor.constant(0.5)).log();
  tensor(0, 0, 0, 0) = 10.0;

  Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> tensor_argmax;

  tensor_argmax = tensor.argmax();

  return 0.0;
#endif

  // reshape action
  Eigen::Tensor<float, 1, Eigen::RowMajor> action_vec = action.shaped<float, 1>({INPUT_SIZE});

  // argmax
  Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> action_note = action_vec.argmax();

  // reshape reward_scores
  Eigen::Tensor<float, 1, Eigen::RowMajor> reward_scores_vec = reward_scores.shaped<float, 1>({INPUT_SIZE});

  // logsumexp
  Eigen::Tensor<float, 0, Eigen::RowMajor> normalization_constant = reward_scores_vec.exp().sum().log();

  return reward_scores_vec(static_cast<int>(action_note())) - normalization_constant();
}

Tensor RLTuner::CollectReward(const Tensor &obs, const  Tensor &action, const Tensor &reward_scores) {
    // Gets and saves log p(a|s) as output by reward_rnn.
    double note_rnn_reward = this->RewardFromRewardRnnScores(action, reward_scores);
    this->note_rnn_reward_last_n += note_rnn_reward;

    double reward = this->RewardMusicTheory(action);

    this->music_theory_reward_last_n += reward * REWARD_SCALER;

    return Tensor(static_cast<float>(reward * REWARD_SCALER + note_rnn_reward));
}


// Music Theory
double RLTuner::RewardMusicTheory(const Tensor &action) {
  double reward = this->RewardKey(action);

  // More MT rewards TODO


  return reward;
}

double RLTuner::RewardKey(const Tensor &action) {
  double penalty_amount = -1.0;

  double reward = 0;

  // reshape action
  Eigen::Tensor<float, 1, Eigen::RowMajor> action_vec = action.shaped<float, 1>({INPUT_SIZE});

  Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> action_note = action_vec.argmax();

  std::unordered_set<int> key(C_MAJOR_KEY);
  if (key.find(static_cast<int>(action_note())) == key.end())
    reward = penalty_amount;

  return reward;
}

}  // namespace tensorflow
