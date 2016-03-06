#pragma once

#include "../common/Common.hpp"
#include <vector>
#include <iosfwd>


struct TrainingSample {
    vector<float> input;
    vector<float> expectedOutput;

    TrainingSample(const vector<float> &input, const vector<float> expectedOutput) :
      input(input), expectedOutput(expectedOutput) {}
};

std::ostream& operator<<(std::ostream& stream, const TrainingSample& ts);

class Network {
public:
  Network(const vector<unsigned> &layerSizes);
  virtual ~Network();

  vector<float> Process(const vector<float> &input);
  void Train(const vector<TrainingSample> &samples, float learnRate);

  std::ostream& Output(std::ostream& stream);

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
