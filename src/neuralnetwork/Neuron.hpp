#pragma once

#include "../common/Common.hpp"


enum class NeuronType {
  INPUT, INTERNAL, OUTPUT,
};


class Neuron {
public:

  Neuron(NeuronType type);
  virtual ~Neuron();

  void AddIncomingNeuron(wptr<Neuron> neuron);
  void AddOutgoingNeuron(wptr<Neuron> neuron);

  void UpdateWeights(float normScale, float rate);
  void UpdateDeltas(void);

  void CalculateOutput(void);

  void SetInput(float input); // called only for input nodes
  void SetError(float error);

  float GetInputWeight(Neuron *incoming) const;
  float GetOutput(void) const; // used for forward propagation
  float GetError(void) const; // used for back propagation

  std::ostream& Output(std::ostream& stream);

private:
  struct NeuronImpl;
  uptr<NeuronImpl> impl;
};
