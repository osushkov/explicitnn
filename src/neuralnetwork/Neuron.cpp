
#include "Neuron.hpp"
#include "../util/Util.hpp"
#include "../common/Common.hpp"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>


struct IncomingConnection {
  bool isBias;

  wptr<Neuron> neuron;
  float weight;

  float weightGradient; // derr/dweight

  IncomingConnection(wptr<Neuron> neuron, float weight) :
      isBias(false), neuron(neuron), weight(weight), weightGradient(0.0f) {}

  IncomingConnection(float weight) :
      isBias(true), weight(weight), weightGradient(0.0f) {}

  float CalculateValue(void) const {
    if (isBias) {
      return weight;
    } else {
      sptr<Neuron> acquiredNeuron = neuron.lock();
      assert(acquiredNeuron);

      return acquiredNeuron->GetOutput() * weight;
    }
  }

  float GetOutput(void) {
    if (isBias) {
      return 1.0;
    } else {
      sptr<Neuron> acquiredNeuron = neuron.lock();
      assert(acquiredNeuron);

      return acquiredNeuron->GetOutput();
    }
  }
};


struct Neuron::NeuronImpl {
  const float INIT_RANGE = 0.1f;

  Neuron *thisNeuron;

  NeuronType type;
  vector<IncomingConnection> incoming;
  vector<wptr<Neuron>> outgoingNeurons;

  float input;
  float output;
  float error;


  NeuronImpl(Neuron *thisNeuron, NeuronType type) : thisNeuron(thisNeuron), type(type) {
    assert(thisNeuron != nullptr);
    incoming.emplace_back(INIT_RANGE); //Util::RandInterval(-INIT_RANGE, INIT_RANGE)); // Add the bias input.
  }

  void AddIncomingNeuron(wptr<Neuron> neuron) {
    assert(neuron.lock());
    incoming.emplace_back(neuron, INIT_RANGE) ;//Util::RandInterval(-INIT_RANGE, INIT_RANGE));
  }

  void AddOutgoingNeuron(wptr<Neuron> neuron) {
    assert(neuron.lock());
    outgoingNeurons.push_back(neuron);
  }

  void UpdateWeights(float normScale, float rate) {
    if (type != NeuronType::INPUT) {
      for_each(incoming, [normScale, rate] (IncomingConnection &connection) {
        float gradient = connection.weightGradient * normScale;
        connection.weight -= gradient * rate;
        connection.weightGradient = 0.0f;
      });
    }
  }

  void UpdateDeltas(void) {
    if (type == NeuronType::INPUT) {
      return;
    }

    // For output neurons the error is set by the network.
    if (type != NeuronType::OUTPUT) {
      this->error = calculateError();
    }

    for (auto& connection : incoming) {
      connection.weightGradient += this->error * connection.GetOutput();
    }

  }

  void CalculateOutput(void) {
    if (type == NeuronType::INPUT) {
      this->output = input;
    } else {
      float z = calculateZ();
      this->output = activationFunction(z);
    }
  }

  void SetInput(float input) {
    assert(type == NeuronType::INPUT);
    this->input = input;
  }

  void SetError(float error) {
    this->error = error;
  }

  float GetInputWeight(Neuron *target) const {
    Maybe<IncomingConnection> connection = find_if(incoming, [target] (const IncomingConnection c) {
      if (c.isBias) {
        return false;
      }
      sptr<Neuron> acquiredNeuron = c.neuron.lock();
      return acquiredNeuron && acquiredNeuron.get() == target;
    });

    assert(connection.valid());
    return connection.val().weight;
  }

  float GetOutput(void) const {
    return output;
  }

  float GetError(void) const {
    return error;
  }


  float calculateError(void) {
    float outErrorSum = 0.0;
    for (auto& connection : outgoingNeurons) {
      sptr<Neuron> neuron = connection.lock();
      assert(neuron);

      outErrorSum += neuron->GetError() * neuron->GetInputWeight(thisNeuron);
    }

    return this->output * (1.0 - this->output) * outErrorSum;
  }

  float calculateZ(void) {
    float z = 0.0f;
    for (const auto& connection : incoming) {
      z += connection.CalculateValue();
    }
    return z;
  }

  float activationFunction(float in) {
    return 1.0f / (1.0f + expf(-in));
  }
};


Neuron::Neuron(NeuronType type) : impl(new NeuronImpl(this, type)) {}
Neuron::~Neuron() = default;

void Neuron::AddIncomingNeuron(wptr<Neuron> neuron) {
  impl->AddIncomingNeuron(neuron);
}

void Neuron::AddOutgoingNeuron(wptr<Neuron> neuron) {
  impl->AddOutgoingNeuron(neuron);
}

void Neuron::UpdateWeights(float normScale, float rate) {
  impl->UpdateWeights(normScale, rate);
}

void Neuron::UpdateDeltas(void) {
  impl->UpdateDeltas();
}

void Neuron::CalculateOutput(void) {
  impl->CalculateOutput();
}

void Neuron::SetInput(float input) {
  impl->SetInput(input);
}

void Neuron::SetError(float error) {
  impl->SetError(error);
}

float Neuron::GetInputWeight(Neuron *incoming) const {
  return impl->GetInputWeight(incoming);
}

float Neuron::GetOutput(void) const {
  return impl->GetOutput();
}

float Neuron::GetError(void) const {
  return impl->GetError();
}

std::ostream& Neuron::Output(std::ostream& stream) {
  for (const auto& c : impl->incoming) {
    stream << c.weight << " ";
  }
  return stream;
}
