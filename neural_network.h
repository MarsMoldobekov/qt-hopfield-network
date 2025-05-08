#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <list>
#include <vector>
#include <cstdint>
#include <stdexcept>

class NeuronNet {
public:
    enum class State : std::int8_t { Lower = -1, Upper = 1 };
    using Pattern = std::vector<State>;

    explicit NeuronNet(std::size_t neuron_count);
    ~NeuronNet() = default;

    void learn(const std::list<Pattern> &patterns);
    std::size_t recognize(Pattern &pattern) const;

    static State read(std::uint8_t value) noexcept;
    static std::uint8_t write(State state) noexcept;

private:
    static constexpr int MAX_STEP = 1000;
    std::size_t neuron_count_;
    std::vector<std::vector<double>> synapses_;

    bool update(Pattern &pattern) const;
    static double multiply(State a, State b) noexcept;
};

#endif // NEURAL_NETWORK_H
