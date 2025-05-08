#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

#include "neural_network.h"

NeuronNet::NeuronNet(std::size_t neuron_count) : neuron_count_(neuron_count) {

}

void NeuronNet::learn(const std::list<Pattern> &patterns) {
    neuron_count_ = patterns.empty() ? 0 : patterns.front().size();

    if (neuron_count_ == 0) {
        throw std::invalid_argument("Pattern list cannot be empty");
    }

    for (const auto &pattern : patterns) {
        if (pattern.size() != neuron_count_) {
            throw std::invalid_argument("All pattern must be same size");
        }
    }

    synapses_.resize(neuron_count_, std::vector<double>(neuron_count_, 0.0));
    const double normalization = 1.0 / neuron_count_;
    for (std::size_t i = 0; i < neuron_count_; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            double sum = 0.0;
            for (const auto &pattern : patterns) {
                sum += multiply(pattern[i], pattern[j]);
            }
            synapses_[i][j] = synapses_[j][i] = sum * normalization;
        }
        synapses_[i][i] = 0.0;
    }
}

double NeuronNet::multiply(State a, State b) noexcept {
    return static_cast<double>(static_cast<std::int8_t>(a)) * static_cast<double>(static_cast<std::int8_t>(b));
}

std::size_t NeuronNet::recognize(Pattern &pattern) const {
    if (pattern.size() != neuron_count_) {
        throw std::invalid_argument("Input pattern size mismatch");
    }

    std::size_t steps = 0;
    while (update(pattern) && (++steps < MAX_STEP));
    return steps;
}

bool NeuronNet::update(Pattern &pattern) const {
    bool changed = false;
    std::vector<std::size_t> indices(neuron_count_);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (std::size_t idx : indices) {
        double activation = 0.0;
        for (std::size_t j = 0; j < neuron_count_; ++j) {
            activation += synapses_[idx][j] * static_cast<double>(static_cast<std::int8_t>(pattern[j]));
        }

        State newState = activation > 0 ? State::Upper : State::Lower;
        if (newState != pattern[idx]) {
            pattern[idx] = newState;
            changed = true;
        }
    }

    return changed;
}

NeuronNet::State NeuronNet::read(std::uint8_t value) noexcept {
    return value == 0 ? State::Upper : State::Lower;
}

std::uint8_t NeuronNet::write(State state) noexcept {
    return state == State::Upper ? 0 : 255;
}
