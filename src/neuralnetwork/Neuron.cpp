
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
  double weight;

  double weightGradient; // derr/dweight

  IncomingConnection(wptr<Neuron> neuron, double weight) :
      isBias(false), neuron(neuron), weight(weight), weightGradient(0.0) {}

  IncomingConnection(double weight) :
      isBias(true), weight(weight), weightGradient(0.0) {}

  double CalculateValue(void) const {
    if (isBias) {
      return weight;
    } else {
      sptr<Neuron> acquiredNeuron = neuron.lock();
      assert(acquiredNeuron);

      return acquiredNeuron->GetOutput() * weight;
    }
  }

  double GetOutput(void) {
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
  const double INIT_RANGE = 0.5;

  Neuron *thisNeuron;

  NeuronType type;
  vector<IncomingConnection> incoming;
  vector<wptr<Neuron>> outgoingNeurons;

  double input;
  double output;
  double error;


  NeuronImpl(Neuron *thisNeuron, NeuronType type) : thisNeuron(thisNeuron), type(type) {
    assert(thisNeuron != nullptr);
    incoming.emplace_back(Util::RandInterval(-INIT_RANGE, INIT_RANGE)); // Add the bias input.
  }

  void AddIncomingNeuron(wptr<Neuron> neuron) {
    assert(neuron.lock());
    incoming.emplace_back(neuron, Util::RandInterval(-INIT_RANGE, INIT_RANGE));
  }

  void AddOutgoingNeuron(wptr<Neuron> neuron) {
    assert(neuron.lock());
    outgoingNeurons.push_back(neuron);
  }

  void UpdateWeights(double normScale, double rate) {
    if (type != NeuronType::INPUT) {
      for_each(incoming, [normScale, rate] (IncomingConnection &connection) {
        double gradient = connection.weightGradient * normScale;
        connection.weight -= gradient * rate;
        connection.weightGradient = 0.0;
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
      double z = calculateZ();
      this->output = activationFunction(z);
    }
  }

  void SetInput(double input) {
    assert(type == NeuronType::INPUT);
    this->input = input;
  }

  void SetError(double error) {
    this->error = error;
  }

  double GetInputWeight(Neuron *target) const {
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

  double GetOutput(void) const {
    return output;
  }

  double GetError(void) const {
    return error;
  }


  double calculateError(void) {
    double outErrorSum = 0.0;
    for (auto& connection : outgoingNeurons) {
      sptr<Neuron> neuron = connection.lock();
      assert(neuron);

      outErrorSum += neuron->GetError() * neuron->GetInputWeight(thisNeuron);
    }

    return this->output * (1.0 - this->output) * outErrorSum;
  }

  double calculateZ(void) {
    double z = 0.0;
    for (const auto& connection : incoming) {
      z += connection.CalculateValue();
    }
    return z;
  }

  double activationFunction(double in) {
    return 1.0 / (1.0 + exp(-in));
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

void Neuron::UpdateWeights(double normScale, double rate) {
  impl->UpdateWeights(normScale, rate);
}

void Neuron::UpdateDeltas(void) {
  impl->UpdateDeltas();
}

void Neuron::CalculateOutput(void) {
  impl->CalculateOutput();
}

void Neuron::SetInput(double input) {
  impl->SetInput(input);
}

void Neuron::SetError(double error) {
  impl->SetError(error);
}

double Neuron::GetInputWeight(Neuron *incoming) const {
  return impl->GetInputWeight(incoming);
}

double Neuron::GetOutput(void) const {
  return impl->GetOutput();
}

double Neuron::GetError(void) const {
  return impl->GetError();
}
