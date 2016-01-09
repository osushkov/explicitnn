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

  void UpdateWeights(double normScale, double rate);
  void UpdateDeltas(void);

  void CalculateOutput(void);

  void SetInput(double input); // called only for input nodes
  void SetError(double error);

  double GetInputWeight(Neuron *incoming) const;
  double GetOutput(void) const; // used for forward propagation
  double GetError(void) const; // used for back propagation

private:
  struct NeuronImpl;
  uptr<NeuronImpl> impl;
};
