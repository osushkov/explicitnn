
#include "Network.hpp"
#include "Neuron.hpp"
#include <algorithm>
#include <cassert>
#include <ostream>
#include <iostream>


struct Layer {
  NeuronType type;
  vector<sptr<Neuron>> neurons;

  Layer(NeuronType type, unsigned size) : type(type) {
    assert(size > 0);
    neurons.reserve(size);

    for (unsigned i = 0; i < size; i++) {
      neurons.emplace_back(new Neuron(type));
    }
  }

  ~Layer() {
    cout << "Layer destructor called" << endl;
  }
};


struct Network::NetworkImpl {
  vector<Layer> layers;


  NetworkImpl(const vector<unsigned> &layerSizes) {
    assert(layerSizes.size() >= 2);
    layers.reserve(layerSizes.size());

    // Create the layers.
    for (unsigned i = 0; i < layerSizes.size(); i++) {
      NeuronType layerType =
        (i == 0) ? NeuronType::INPUT :
        (i == layerSizes.size() - 1) ? NeuronType::OUTPUT :
        NeuronType::INTERNAL;

      layers.emplace_back(layerType, layerSizes[i]);
    }

    // Connect the layers.
    for (unsigned i = 0; i < layers.size()-1; i++) {
      connectLayers(layers[i], layers[i+1]);
    }
  }

  static void connectLayers(Layer &layer0, Layer &layer1) {
    for (auto& neuron0 : layer0.neurons) {
      for (auto& neuron1 : layer1.neurons) {
        neuron0->AddOutgoingNeuron(neuron1);
        neuron1->AddIncomingNeuron(neuron0);
      }
    }
  }

  vector<float> Process(const vector<float> &input) {
    vector<float> result;

    for (auto& layer : layers) {
      if (layer.type == NeuronType::INPUT) {
        assert(layer.neurons.size() == input.size());

        for (unsigned i = 0; i < layer.neurons.size(); i++) {
          layer.neurons[i]->SetInput(input[i]);
        }
      }

      for_each(layer.neurons, [](sptr<Neuron> neuron) { neuron->CalculateOutput(); });

      if (layer.type == NeuronType::OUTPUT) {
        assert(result.empty());

        result.reserve(layer.neurons.size());
        for (auto& neuron : layer.neurons) {
          result.push_back(neuron->GetOutput());
        }
      }
    }

    return result;
  }

  void Train(const vector<TrainingSample> &samples, float learnRate) {
    for (const auto &sample : samples) {
      processSample(sample);
    }

    float normScale = 1.0f / samples.size();
    for (auto& layer : layers) {
      if (layer.type != NeuronType::INPUT) {
        for_each(layer.neurons, [=] (sptr<Neuron> neuron) {
          neuron->UpdateWeights(normScale, learnRate);
        });
      }
    }
  }

  void processSample(const TrainingSample &sample) {
    vector<float> output = Process(sample.input);
    Layer &outputLayer = layers[layers.size() - 1];

    assert(outputLayer.type == NeuronType::OUTPUT);
    assert(outputLayer.neurons.size() == output.size());
    assert(output.size() == sample.expectedOutput.size());

    for (unsigned i = 0; i < output.size(); i++) {
      float o = output[i];
      // float delta = o * (1.0 - o) * (o - sample.expectedOutput[i]);
      float delta = (o - sample.expectedOutput[i]);
      outputLayer.neurons[i]->SetError(delta);
    }

    for (unsigned i = layers.size() - 1; i >= 1; i--) {
      for_each(layers[i].neurons, [](sptr<Neuron> neuron) { neuron->UpdateDeltas(); });
    }
  }
};


Network::Network(const vector<unsigned> &layerSizes) : impl(new NetworkImpl(layerSizes)) {}
Network::~Network() = default;

vector<float> Network::Process(const vector<float> &input) {
  return impl->Process(input);
}

void Network::Train(const vector<TrainingSample> &samples, float learnRate) {
  assert(learnRate >= 0.0f && learnRate <= 1.0f);
  impl->Train(samples, learnRate);
}

std::ostream& Network::Output(std::ostream& stream) {
  for (auto& layer : impl->layers) {
    if (layer.type != NeuronType::INPUT) {
      for (auto& n : layer.neurons) {
        n->Output(stream);
        stream << endl;
      }
      stream << endl;
    }
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TrainingSample& ts) {
  for (auto i : ts.input) {
    stream << i << "\t";
  }
  stream << ": ";
  for (auto eo : ts.expectedOutput) {
    stream << eo << "\t";
  }
  return stream;
}
