#pragma once

#include <cmath>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>


//-----------------------------------------------------------------------------
// VALUE
//-----------------------------------------------------------------------------
namespace tg
{
    struct Value
    {
        size_t id;
        explicit Value(const size_t id) : id(id) {}
    };
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// OPERATORS
//-----------------------------------------------------------------------------
namespace tg
{
    class Tigrad;
    namespace ops
    {
        struct Add
        {
            size_t out, a, b;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "+"; }
        };

        struct Sub
        {
            size_t out, a, b;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "-"; }
        };

        struct Mul
        {
            size_t out, a, b;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "*"; }
        };

        struct Div
        {
            size_t out, a, b;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "/"; }
        };

        struct Pow
        {
            size_t out, a, b;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "^"; }
        };

        struct Exp
        {
            size_t out, a;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "exp"; }
        };

        struct Tanh
        {
            size_t out, a;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "tanh"; }
        };

        struct Relu
        {
            size_t out, a;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "relu"; }
        };

        struct Sigmoid
        {
            size_t out, a;
            void calcData(Tigrad* ctx) const;
            void calcGrad(Tigrad* ctx) const;
            static std::string symbol() { return "sigmoid"; }
        };

        using OpType = std::variant<Add, Sub, Mul, Div, Pow, Exp, Tanh, Relu, Sigmoid>;
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// TIGRAD ENGINE
//-----------------------------------------------------------------------------
namespace tg
{
    class Tigrad
    {
    public:
        Tigrad() = default;
        ~Tigrad() = default;
        Tigrad(const Tigrad&) = delete;
        Tigrad& operator=(const Tigrad&) = delete;
        Tigrad(Tigrad&&) = delete;
        Tigrad& operator=(Tigrad&&) = delete;

        static void setActive(Tigrad* ctx) { _sActive = ctx; }
        static Tigrad* getActive() { return _sActive; }

        Value createVal();
        Value createVal(const std::string& valLabel);
        Value createConstant(float val);
        void pushOp(const ops::OpType &opType);

        static void visualize(Tigrad* ctx, const std::string& filename = "output.dot");


        //-----------------------------------------------------------------------------
        // ACCESSOR UTILS
        //-----------------------------------------------------------------------------
        struct DataAccessor
        {
            Tigrad* ctx;
            // NOLINTNEXTLINE(readability-make-member-function-const)
            float& operator[](const Value val) { return ctx->_data[val.id]; }
            float operator[](const Value val) const { return ctx->_data[val.id]; }
            // NOLINTNEXTLINE(readability-make-member-function-const)
            float& operator[](const size_t idx) { return ctx->_data[idx]; }
            float operator[](const size_t idx) const { return ctx->_data[idx]; }
        };

        struct GradAccessor
        {
            Tigrad* ctx;
            // NOLINTNEXTLINE(readability-make-member-function-const)
            float& operator[](const Value val) { return ctx->_grad[val.id]; }
            float operator[](const Value val) const { return ctx->_grad[val.id]; }
            // NOLINTNEXTLINE(readability-make-member-function-const)
            float& operator[](const size_t idx) { return ctx->_grad[idx]; }
            float operator[](const size_t idx) const { return ctx->_grad[idx]; }
        };

        struct LabelAccessor
        {
            Tigrad* ctx;
            // NOLINTNEXTLINE(readability-make-member-function-const)
            std::string& operator[](const Value val) { return ctx->_labels[val.id]; }
            std::string operator[](const Value val) const { return ctx->_labels[val.id]; }
            // NOLINTNEXTLINE(readability-make-member-function-const)
            std::string& operator[](const size_t idx) { return ctx->_labels[idx]; }
            std::string operator[](const size_t idx) const { return ctx->_labels[idx]; }
        };

        std::vector<ops::OpType> getOpList() { return _opList; }

        DataAccessor data { this };
        GradAccessor grad { this };
        LabelAccessor label { this };
        //-----------------------------------------------------------------------------


        //-----------------------------------------------------------------------------
        // CORE FUNCTIONS
        //-----------------------------------------------------------------------------
        size_t compile();
        void calcData();
        void calcGrad(Value outputNode);
        void zeroGrad();
        //-----------------------------------------------------------------------------

    private:
        static inline Tigrad* _sActive = nullptr;
        size_t _nextIdx = 0;
        bool _compiled = false;
        std::vector<std::string> _labels;
        std::vector<float> _data;
        std::vector<float> _grad;
        std::vector<std::pair<int, float>> _constants;
        std::vector<ops::OpType> _opList;
    };
}
//-----------------------------------------------------------------------------



#ifdef TIGRAD_IMPLEMENTATION

namespace tg
{
    // ADD OP
    inline Value operator+(const Value& lhs, const Value& rhs)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Add{out.id, lhs.id, rhs.id});
        return out;
    }
    inline Value operator+(const Value& lhs, const float rhs) { return lhs + Tigrad::getActive()->createConstant(rhs); }
    inline Value operator+(const float lhs, const Value& rhs) { return Tigrad::getActive()->createConstant(lhs) + rhs; }
    inline void ops::Add::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = ctx->data[a] + ctx->data[b];
    }
    inline void ops::Add::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += ctx->grad[out];
        ctx->grad[b] += ctx->grad[out];
    }

    // SUB OP
    inline Value operator-(const Value& lhs, const Value& rhs)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Sub{out.id, lhs.id, rhs.id});
        return out;
    }
    inline Value operator-(const Value& lhs, const float rhs) { return lhs - Tigrad::getActive()->createConstant(rhs); }
    inline Value operator-(const float lhs, const Value& rhs) { return Tigrad::getActive()->createConstant(lhs) - rhs; }
    inline void ops::Sub::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = ctx->data[a] - ctx->data[b];
    }
    inline void ops::Sub::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += ctx->grad[out];
        ctx->grad[b] += -ctx->grad[out];
    }

    // MUL OP
    inline Value operator*(const Value& lhs, const Value& rhs)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Mul{out.id, lhs.id, rhs.id});
        return out;
    }
    inline Value operator*(const Value& lhs, const float rhs) { return lhs * Tigrad::getActive()->createConstant(rhs); }
    inline Value operator*(const float lhs, const Value& rhs) { return Tigrad::getActive()->createConstant(lhs) * rhs; }
    inline void ops::Mul::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = ctx->data[a] * ctx->data[b];
    }
    inline void ops::Mul::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += ctx->data[b] * ctx->grad[out];
        ctx->grad[b] += ctx->data[a] * ctx->grad[out];
    }

    // DIV OP
    inline Value operator/(const Value& lhs, const Value& rhs)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Div{out.id, lhs.id, rhs.id});
        return out;
    }
    inline Value operator/(const Value& lhs, const float rhs) { return lhs / Tigrad::getActive()->createConstant(rhs); }
    inline Value operator/(const float lhs, const Value& rhs) { return Tigrad::getActive()->createConstant(lhs) / rhs; }
    inline void ops::Div::calcData(Tigrad *ctx) const
    {
        if (ctx->data[b] == 0.0f)
            throw std::runtime_error("Division by zero!");
        ctx->data[out] = ctx->data[a] / ctx->data[b];
    }
    inline void ops::Div::calcGrad(Tigrad *ctx) const
    {
        if (ctx->data[b] == 0.0f)
            throw std::runtime_error("Division by zero!");

        const float invRhs = 1.0f / ctx->data[b];
        ctx->grad[a] += invRhs * ctx->grad[out];
        ctx->grad[b] += (-ctx->data[a] * (invRhs * invRhs)) * ctx->grad[out];
    }

    // POW OP
    inline Value pow(const Value& base, const float exp)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        const Value exponent = ctx->createConstant(exp);
        ctx->pushOp(ops::Pow(out.id, base.id, exponent.id));
        return out;
    }
    inline void ops::Pow::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = std::pow(ctx->data[a], ctx->data[b]);
    }
    inline void ops::Pow::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += ctx->data[b] * std::pow(ctx->data[a], ctx->data[b] - 1.0f) * ctx->grad[out];
    }

    // EXP OP
    inline Value exp(const Value& power)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Exp(out.id, power.id));
        return out;
    }
    inline void ops::Exp::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = std::exp(ctx->data[a]);
    }
    inline void ops::Exp::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += std::exp(ctx->data[a]) * ctx->grad[out];
    }

    // TANH OP
    inline Value tanh(const Value& arg)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Tanh(out.id, arg.id));
        return out;
    }
    inline void ops::Tanh::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = std::tanh(ctx->data[a]);
    }
    inline void ops::Tanh::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += (1.0f - (ctx->data[out] * ctx->data[out])) * ctx->grad[out];
    }

    // RELU OP
    inline Value relu(const Value& arg)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Relu(out.id, arg.id));
        return out;
    }
    inline void ops::Relu::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = std::max(0.0f, ctx->data[a]);
    }
    inline void ops::Relu::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += static_cast<float>(ctx->data[out] > 0.0f) * ctx->grad[out];
    }

    // SIGMOID OP
    inline Value sigmoid(const Value& arg)
    {
        Tigrad* ctx = Tigrad::getActive();
        const Value out = ctx->createVal();
        ctx->pushOp(ops::Sigmoid(out.id, arg.id));
        return out;
    }
    inline void ops::Sigmoid::calcData(Tigrad *ctx) const
    {
        ctx->data[out] = 1.0f / (1.0f + std::exp(-ctx->data[a]));
    }
    inline void ops::Sigmoid::calcGrad(Tigrad *ctx) const
    {
        ctx->grad[a] += (ctx->data[out] * (1.0f - ctx->data[out])) * ctx->grad[out];
    }
}

namespace tg
{
    inline Value Tigrad::createVal()
    {
        if (_compiled) { throw std::runtime_error("Operations not permitted after calling compile()"); }
        _labels.push_back("t" + std::to_string(_nextIdx));
        return Value(_nextIdx++);
    }

    inline Value Tigrad::createVal(const std::string &valLabel)
    {
        if (_compiled) { throw std::runtime_error("Operations not permitted after calling compile()"); }
        _labels.push_back(valLabel);
        return Value(_nextIdx++);
    }

    inline Value Tigrad::createConstant(float val)
    {
        Value v = createVal();
        _constants.emplace_back(v.id, val);
        return v;
    }

    inline void Tigrad::pushOp(const ops::OpType &opType)
    {
        _opList.push_back(opType);
    }

    inline size_t Tigrad::compile()
    {
        if (_compiled) { throw std::runtime_error("Already compiled once!"); }
        _data.assign(_nextIdx, 0.0f);
        _grad.assign(_nextIdx, 0.0f);

        for (auto& [idx, val] : _constants)
            _data.at(idx) = val;

        _compiled = true;
        return _nextIdx;
    }

    inline void Tigrad::calcData()
    {
        for (const auto& op : _opList)
        {
            std::visit([this](const auto& specificOp)
            {
                specificOp.calcData(this);
            }, op);
        }
    }

    inline void Tigrad::calcGrad(const Value outputNode)
    {
        _grad.at(outputNode.id) = 1.0f;
        for (const auto& op : std::views::reverse(_opList))
        {
            std::visit([this](const auto& specificOp)
            {
                specificOp.calcGrad(this);
            }, op);
        }
    }

    inline void Tigrad::zeroGrad()
    {
        if (!_compiled) { throw std::runtime_error("Operation requires calling compile() first!"); }
        std::fill(_grad.begin(), _grad.end(), 0.0f);
    }

    inline void Tigrad::visualize(Tigrad* ctx, const std::string& filename)
    {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Failed to open file: " + filename);

        std::unordered_set<size_t> emittedNodes;
        size_t opCnt = 0;

        auto emitNode = [&](const size_t id)
        {
            if (!emittedNodes.insert(id).second) return;
            file << "val" << id << " [label=\"{ " << ctx->label[id]
                 << " | data " << std::to_string(ctx->data[id])
                 << " | grad " << std::to_string(ctx->grad[id]) << " }\"];\n";
        };

        file << "digraph G {\n";
        file << "rankdir=LR\n";
        file << "node [shape=record, fontname=\"Helvetica\"]\n";
        for (const auto& op : ctx->getOpList())
        {
            std::visit([&](const auto& specificOp)
            {
                const std::string opName = "op" + std::to_string(opCnt++);
                file << opName << " [label=\"" << specificOp.symbol() << "\"];\n";

                emitNode(specificOp.a);
                file << "val" << specificOp.a << " -> " << opName << "\n";

                if constexpr (requires { specificOp.b; })
                {
                    emitNode(specificOp.b);
                    file << "val" << specificOp.b << " -> " << opName << "\n";
                }

                emitNode(specificOp.out);
                file << opName << " -> val" << specificOp.out << "\n";
            }, op);
        }
        file << "\n}";
    }
} // namespace tg

#endif
