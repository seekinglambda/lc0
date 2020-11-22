// This file is AUTOGENERATED, do not edit.
#pragma once
#include "utils/protomessage.h"
namespace pblczero {
  class EngineVersion : public lczero::ProtoMessage {
   public:

    bool has_major() const { return has_major_; }
    std::uint32_t major() const { return major_; }
    void set_major(std::uint32_t val) {
      has_major_ = true;
      major_ = val;
    }

    bool has_minor() const { return has_minor_; }
    std::uint32_t minor() const { return minor_; }
    void set_minor(std::uint32_t val) {
      has_minor_ = true;
      minor_ = val;
    }

    bool has_patch() const { return has_patch_; }
    std::uint32_t patch() const { return patch_; }
    void set_patch(std::uint32_t val) {
      has_patch_ = true;
      patch_ = val;
    }

    void Clear() override {
      has_major_ = false;
      major_ = {};
      has_minor_ = false;
      minor_ = {};
      has_patch_ = false;
      patch_ = {};
    }

   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_major(static_cast<std::uint32_t>(val)); break;
        case 2: set_minor(static_cast<std::uint32_t>(val)); break;
        case 3: set_patch(static_cast<std::uint32_t>(val)); break;
      }
    }

    bool has_major_{};
    std::uint32_t major_{};
    bool has_minor_{};
    std::uint32_t minor_{};
    bool has_patch_{};
    std::uint32_t patch_{};
  };
  class Weights : public lczero::ProtoMessage {
   public:
    class Layer : public lczero::ProtoMessage {
     public:

      bool has_min_val() const { return has_min_val_; }
      float min_val() const { return min_val_; }
      void set_min_val(float val) {
        has_min_val_ = true;
        min_val_ = val;
      }

      bool has_max_val() const { return has_max_val_; }
      float max_val() const { return max_val_; }
      void set_max_val(float val) {
        has_max_val_ = true;
        max_val_ = val;
      }

      bool has_params() const { return has_params_; }
      std::string_view params() const { return params_; }
      void set_params(std::string_view val) {
        has_params_ = true;
        params_ = val;
      }

      void Clear() override {
        has_min_val_ = false;
        min_val_ = {};
        has_max_val_ = false;
        max_val_ = {};
        has_params_ = false;
        params_ = {};
      }

     private:
      void SetInt32(int field_id, std::uint32_t val) override {
        switch (field_id) {
          case 1: set_min_val(bit_cast<float>(val)); break;
          case 2: set_max_val(bit_cast<float>(val)); break;
        }
      }
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 3: set_params(val); break;
        }
      }

      bool has_min_val_{};
      float min_val_{};
      bool has_max_val_{};
      float max_val_{};
      bool has_params_{};
      std::string params_{};
    };
    class ConvBlock : public lczero::ProtoMessage {
     public:

      bool has_weights() const { return has_weights_; }
      const Layer& weights() const { return weights_; }
      Layer* mutable_weights() {
        has_weights_ = true;
        return &weights_;
      }

      bool has_biases() const { return has_biases_; }
      const Layer& biases() const { return biases_; }
      Layer* mutable_biases() {
        has_biases_ = true;
        return &biases_;
      }

      bool has_bn_means() const { return has_bn_means_; }
      const Layer& bn_means() const { return bn_means_; }
      Layer* mutable_bn_means() {
        has_bn_means_ = true;
        return &bn_means_;
      }

      bool has_bn_stddivs() const { return has_bn_stddivs_; }
      const Layer& bn_stddivs() const { return bn_stddivs_; }
      Layer* mutable_bn_stddivs() {
        has_bn_stddivs_ = true;
        return &bn_stddivs_;
      }

      bool has_bn_gammas() const { return has_bn_gammas_; }
      const Layer& bn_gammas() const { return bn_gammas_; }
      Layer* mutable_bn_gammas() {
        has_bn_gammas_ = true;
        return &bn_gammas_;
      }

      bool has_bn_betas() const { return has_bn_betas_; }
      const Layer& bn_betas() const { return bn_betas_; }
      Layer* mutable_bn_betas() {
        has_bn_betas_ = true;
        return &bn_betas_;
      }

      void Clear() override {
        has_weights_ = false;
        weights_ = {};
        has_biases_ = false;
        biases_ = {};
        has_bn_means_ = false;
        bn_means_ = {};
        has_bn_stddivs_ = false;
        bn_stddivs_ = {};
        has_bn_gammas_ = false;
        bn_gammas_ = {};
        has_bn_betas_ = false;
        bn_betas_ = {};
      }

     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_weights()->MergeFromString(val); break;
          case 2: mutable_biases()->MergeFromString(val); break;
          case 3: mutable_bn_means()->MergeFromString(val); break;
          case 4: mutable_bn_stddivs()->MergeFromString(val); break;
          case 5: mutable_bn_gammas()->MergeFromString(val); break;
          case 6: mutable_bn_betas()->MergeFromString(val); break;
        }
      }

      bool has_weights_{};
      Layer weights_{};
      bool has_biases_{};
      Layer biases_{};
      bool has_bn_means_{};
      Layer bn_means_{};
      bool has_bn_stddivs_{};
      Layer bn_stddivs_{};
      bool has_bn_gammas_{};
      Layer bn_gammas_{};
      bool has_bn_betas_{};
      Layer bn_betas_{};
    };
    class SEunit : public lczero::ProtoMessage {
     public:

      bool has_w1() const { return has_w1_; }
      const Layer& w1() const { return w1_; }
      Layer* mutable_w1() {
        has_w1_ = true;
        return &w1_;
      }

      bool has_b1() const { return has_b1_; }
      const Layer& b1() const { return b1_; }
      Layer* mutable_b1() {
        has_b1_ = true;
        return &b1_;
      }

      bool has_w2() const { return has_w2_; }
      const Layer& w2() const { return w2_; }
      Layer* mutable_w2() {
        has_w2_ = true;
        return &w2_;
      }

      bool has_b2() const { return has_b2_; }
      const Layer& b2() const { return b2_; }
      Layer* mutable_b2() {
        has_b2_ = true;
        return &b2_;
      }

      void Clear() override {
        has_w1_ = false;
        w1_ = {};
        has_b1_ = false;
        b1_ = {};
        has_w2_ = false;
        w2_ = {};
        has_b2_ = false;
        b2_ = {};
      }

     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_w1()->MergeFromString(val); break;
          case 2: mutable_b1()->MergeFromString(val); break;
          case 3: mutable_w2()->MergeFromString(val); break;
          case 4: mutable_b2()->MergeFromString(val); break;
        }
      }

      bool has_w1_{};
      Layer w1_{};
      bool has_b1_{};
      Layer b1_{};
      bool has_w2_{};
      Layer w2_{};
      bool has_b2_{};
      Layer b2_{};
    };
    class Residual : public lczero::ProtoMessage {
     public:

      bool has_conv1() const { return has_conv1_; }
      const ConvBlock& conv1() const { return conv1_; }
      ConvBlock* mutable_conv1() {
        has_conv1_ = true;
        return &conv1_;
      }

      bool has_conv2() const { return has_conv2_; }
      const ConvBlock& conv2() const { return conv2_; }
      ConvBlock* mutable_conv2() {
        has_conv2_ = true;
        return &conv2_;
      }

      bool has_se() const { return has_se_; }
      const SEunit& se() const { return se_; }
      SEunit* mutable_se() {
        has_se_ = true;
        return &se_;
      }

      void Clear() override {
        has_conv1_ = false;
        conv1_ = {};
        has_conv2_ = false;
        conv2_ = {};
        has_se_ = false;
        se_ = {};
      }

     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_conv1()->MergeFromString(val); break;
          case 2: mutable_conv2()->MergeFromString(val); break;
          case 3: mutable_se()->MergeFromString(val); break;
        }
      }

      bool has_conv1_{};
      ConvBlock conv1_{};
      bool has_conv2_{};
      ConvBlock conv2_{};
      bool has_se_{};
      SEunit se_{};
    };

    bool has_input() const { return has_input_; }
    const ConvBlock& input() const { return input_; }
    ConvBlock* mutable_input() {
      has_input_ = true;
      return &input_;
    }

    Residual* add_residual() { return &residual_.emplace_back(); }
    const std::vector<Residual>& residual() const { return residual_; }
    const Residual& residual(size_t idx) const { return residual_[idx]; }
    size_t residual_size() const { return residual_.size(); }

    bool has_policy1() const { return has_policy1_; }
    const ConvBlock& policy1() const { return policy1_; }
    ConvBlock* mutable_policy1() {
      has_policy1_ = true;
      return &policy1_;
    }

    bool has_policy() const { return has_policy_; }
    const ConvBlock& policy() const { return policy_; }
    ConvBlock* mutable_policy() {
      has_policy_ = true;
      return &policy_;
    }

    bool has_ip_pol_w() const { return has_ip_pol_w_; }
    const Layer& ip_pol_w() const { return ip_pol_w_; }
    Layer* mutable_ip_pol_w() {
      has_ip_pol_w_ = true;
      return &ip_pol_w_;
    }

    bool has_ip_pol_b() const { return has_ip_pol_b_; }
    const Layer& ip_pol_b() const { return ip_pol_b_; }
    Layer* mutable_ip_pol_b() {
      has_ip_pol_b_ = true;
      return &ip_pol_b_;
    }

    bool has_value() const { return has_value_; }
    const ConvBlock& value() const { return value_; }
    ConvBlock* mutable_value() {
      has_value_ = true;
      return &value_;
    }

    bool has_ip1_val_w() const { return has_ip1_val_w_; }
    const Layer& ip1_val_w() const { return ip1_val_w_; }
    Layer* mutable_ip1_val_w() {
      has_ip1_val_w_ = true;
      return &ip1_val_w_;
    }

    bool has_ip1_val_b() const { return has_ip1_val_b_; }
    const Layer& ip1_val_b() const { return ip1_val_b_; }
    Layer* mutable_ip1_val_b() {
      has_ip1_val_b_ = true;
      return &ip1_val_b_;
    }

    bool has_ip2_val_w() const { return has_ip2_val_w_; }
    const Layer& ip2_val_w() const { return ip2_val_w_; }
    Layer* mutable_ip2_val_w() {
      has_ip2_val_w_ = true;
      return &ip2_val_w_;
    }

    bool has_ip2_val_b() const { return has_ip2_val_b_; }
    const Layer& ip2_val_b() const { return ip2_val_b_; }
    Layer* mutable_ip2_val_b() {
      has_ip2_val_b_ = true;
      return &ip2_val_b_;
    }

    bool has_moves_left() const { return has_moves_left_; }
    const ConvBlock& moves_left() const { return moves_left_; }
    ConvBlock* mutable_moves_left() {
      has_moves_left_ = true;
      return &moves_left_;
    }

    bool has_ip1_mov_w() const { return has_ip1_mov_w_; }
    const Layer& ip1_mov_w() const { return ip1_mov_w_; }
    Layer* mutable_ip1_mov_w() {
      has_ip1_mov_w_ = true;
      return &ip1_mov_w_;
    }

    bool has_ip1_mov_b() const { return has_ip1_mov_b_; }
    const Layer& ip1_mov_b() const { return ip1_mov_b_; }
    Layer* mutable_ip1_mov_b() {
      has_ip1_mov_b_ = true;
      return &ip1_mov_b_;
    }

    bool has_ip2_mov_w() const { return has_ip2_mov_w_; }
    const Layer& ip2_mov_w() const { return ip2_mov_w_; }
    Layer* mutable_ip2_mov_w() {
      has_ip2_mov_w_ = true;
      return &ip2_mov_w_;
    }

    bool has_ip2_mov_b() const { return has_ip2_mov_b_; }
    const Layer& ip2_mov_b() const { return ip2_mov_b_; }
    Layer* mutable_ip2_mov_b() {
      has_ip2_mov_b_ = true;
      return &ip2_mov_b_;
    }

    void Clear() override {
      has_input_ = false;
      input_ = {};
      residual_.clear();
      has_policy1_ = false;
      policy1_ = {};
      has_policy_ = false;
      policy_ = {};
      has_ip_pol_w_ = false;
      ip_pol_w_ = {};
      has_ip_pol_b_ = false;
      ip_pol_b_ = {};
      has_value_ = false;
      value_ = {};
      has_ip1_val_w_ = false;
      ip1_val_w_ = {};
      has_ip1_val_b_ = false;
      ip1_val_b_ = {};
      has_ip2_val_w_ = false;
      ip2_val_w_ = {};
      has_ip2_val_b_ = false;
      ip2_val_b_ = {};
      has_moves_left_ = false;
      moves_left_ = {};
      has_ip1_mov_w_ = false;
      ip1_mov_w_ = {};
      has_ip1_mov_b_ = false;
      ip1_mov_b_ = {};
      has_ip2_mov_w_ = false;
      ip2_mov_w_ = {};
      has_ip2_mov_b_ = false;
      ip2_mov_b_ = {};
    }

   private:
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 1: mutable_input()->MergeFromString(val); break;
        case 2: add_residual()->MergeFromString(val); break;
        case 11: mutable_policy1()->MergeFromString(val); break;
        case 3: mutable_policy()->MergeFromString(val); break;
        case 4: mutable_ip_pol_w()->MergeFromString(val); break;
        case 5: mutable_ip_pol_b()->MergeFromString(val); break;
        case 6: mutable_value()->MergeFromString(val); break;
        case 7: mutable_ip1_val_w()->MergeFromString(val); break;
        case 8: mutable_ip1_val_b()->MergeFromString(val); break;
        case 9: mutable_ip2_val_w()->MergeFromString(val); break;
        case 10: mutable_ip2_val_b()->MergeFromString(val); break;
        case 12: mutable_moves_left()->MergeFromString(val); break;
        case 13: mutable_ip1_mov_w()->MergeFromString(val); break;
        case 14: mutable_ip1_mov_b()->MergeFromString(val); break;
        case 15: mutable_ip2_mov_w()->MergeFromString(val); break;
        case 16: mutable_ip2_mov_b()->MergeFromString(val); break;
      }
    }

    bool has_input_{};
    ConvBlock input_{};
    std::vector<Residual> residual_;
    bool has_policy1_{};
    ConvBlock policy1_{};
    bool has_policy_{};
    ConvBlock policy_{};
    bool has_ip_pol_w_{};
    Layer ip_pol_w_{};
    bool has_ip_pol_b_{};
    Layer ip_pol_b_{};
    bool has_value_{};
    ConvBlock value_{};
    bool has_ip1_val_w_{};
    Layer ip1_val_w_{};
    bool has_ip1_val_b_{};
    Layer ip1_val_b_{};
    bool has_ip2_val_w_{};
    Layer ip2_val_w_{};
    bool has_ip2_val_b_{};
    Layer ip2_val_b_{};
    bool has_moves_left_{};
    ConvBlock moves_left_{};
    bool has_ip1_mov_w_{};
    Layer ip1_mov_w_{};
    bool has_ip1_mov_b_{};
    Layer ip1_mov_b_{};
    bool has_ip2_mov_w_{};
    Layer ip2_mov_w_{};
    bool has_ip2_mov_b_{};
    Layer ip2_mov_b_{};
  };
  class TrainingParams : public lczero::ProtoMessage {
   public:

    bool has_training_steps() const { return has_training_steps_; }
    std::uint32_t training_steps() const { return training_steps_; }
    void set_training_steps(std::uint32_t val) {
      has_training_steps_ = true;
      training_steps_ = val;
    }

    bool has_learning_rate() const { return has_learning_rate_; }
    float learning_rate() const { return learning_rate_; }
    void set_learning_rate(float val) {
      has_learning_rate_ = true;
      learning_rate_ = val;
    }

    bool has_mse_loss() const { return has_mse_loss_; }
    float mse_loss() const { return mse_loss_; }
    void set_mse_loss(float val) {
      has_mse_loss_ = true;
      mse_loss_ = val;
    }

    bool has_policy_loss() const { return has_policy_loss_; }
    float policy_loss() const { return policy_loss_; }
    void set_policy_loss(float val) {
      has_policy_loss_ = true;
      policy_loss_ = val;
    }

    bool has_accuracy() const { return has_accuracy_; }
    float accuracy() const { return accuracy_; }
    void set_accuracy(float val) {
      has_accuracy_ = true;
      accuracy_ = val;
    }

    bool has_lc0_params() const { return has_lc0_params_; }
    std::string_view lc0_params() const { return lc0_params_; }
    void set_lc0_params(std::string_view val) {
      has_lc0_params_ = true;
      lc0_params_ = val;
    }

    void Clear() override {
      has_training_steps_ = false;
      training_steps_ = {};
      has_learning_rate_ = false;
      learning_rate_ = {};
      has_mse_loss_ = false;
      mse_loss_ = {};
      has_policy_loss_ = false;
      policy_loss_ = {};
      has_accuracy_ = false;
      accuracy_ = {};
      has_lc0_params_ = false;
      lc0_params_ = {};
    }

   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_training_steps(static_cast<std::uint32_t>(val)); break;
      }
    }
    void SetInt32(int field_id, std::uint32_t val) override {
      switch (field_id) {
        case 2: set_learning_rate(bit_cast<float>(val)); break;
        case 3: set_mse_loss(bit_cast<float>(val)); break;
        case 4: set_policy_loss(bit_cast<float>(val)); break;
        case 5: set_accuracy(bit_cast<float>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 6: set_lc0_params(val); break;
      }
    }

    bool has_training_steps_{};
    std::uint32_t training_steps_{};
    bool has_learning_rate_{};
    float learning_rate_{};
    bool has_mse_loss_{};
    float mse_loss_{};
    bool has_policy_loss_{};
    float policy_loss_{};
    bool has_accuracy_{};
    float accuracy_{};
    bool has_lc0_params_{};
    std::string lc0_params_{};
  };
  class NetworkFormat : public lczero::ProtoMessage {
   public:
    enum InputFormat {
      INPUT_UNKNOWN = 0,
      INPUT_CLASSICAL_112_PLANE = 1,
      INPUT_112_WITH_CASTLING_PLANE = 2,
      INPUT_112_WITH_CANONICALIZATION = 3,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES = 4,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON = 132,
      INPUT_112_WITH_CANONICALIZATION_V2 = 5,
      INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON = 133,
    };
    enum OutputFormat {
      OUTPUT_UNKNOWN = 0,
      OUTPUT_CLASSICAL = 1,
      OUTPUT_WDL = 2,
    };
    enum NetworkStructure {
      NETWORK_UNKNOWN = 0,
      NETWORK_CLASSICAL = 1,
      NETWORK_SE = 2,
      NETWORK_CLASSICAL_WITH_HEADFORMAT = 3,
      NETWORK_SE_WITH_HEADFORMAT = 4,
    };
    enum PolicyFormat {
      POLICY_UNKNOWN = 0,
      POLICY_CLASSICAL = 1,
      POLICY_CONVOLUTION = 2,
    };
    enum ValueFormat {
      VALUE_UNKNOWN = 0,
      VALUE_CLASSICAL = 1,
      VALUE_WDL = 2,
      VALUE_PARAM = 3,
    };
    enum MovesLeftFormat {
      MOVES_LEFT_NONE = 0,
      MOVES_LEFT_V1 = 1,
    };

    bool has_input() const { return has_input_; }
    InputFormat input() const { return input_; }
    void set_input(InputFormat val) {
      has_input_ = true;
      input_ = val;
    }

    bool has_output() const { return has_output_; }
    OutputFormat output() const { return output_; }
    void set_output(OutputFormat val) {
      has_output_ = true;
      output_ = val;
    }

    bool has_network() const { return has_network_; }
    NetworkStructure network() const { return network_; }
    void set_network(NetworkStructure val) {
      has_network_ = true;
      network_ = val;
    }

    bool has_policy() const { return has_policy_; }
    PolicyFormat policy() const { return policy_; }
    void set_policy(PolicyFormat val) {
      has_policy_ = true;
      policy_ = val;
    }

    bool has_value() const { return has_value_; }
    ValueFormat value() const { return value_; }
    void set_value(ValueFormat val) {
      has_value_ = true;
      value_ = val;
    }

    bool has_moves_left() const { return has_moves_left_; }
    MovesLeftFormat moves_left() const { return moves_left_; }
    void set_moves_left(MovesLeftFormat val) {
      has_moves_left_ = true;
      moves_left_ = val;
    }

    void Clear() override {
      has_input_ = false;
      input_ = {};
      has_output_ = false;
      output_ = {};
      has_network_ = false;
      network_ = {};
      has_policy_ = false;
      policy_ = {};
      has_value_ = false;
      value_ = {};
      has_moves_left_ = false;
      moves_left_ = {};
    }

   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_input(static_cast<InputFormat>(val)); break;
        case 2: set_output(static_cast<OutputFormat>(val)); break;
        case 3: set_network(static_cast<NetworkStructure>(val)); break;
        case 4: set_policy(static_cast<PolicyFormat>(val)); break;
        case 5: set_value(static_cast<ValueFormat>(val)); break;
        case 6: set_moves_left(static_cast<MovesLeftFormat>(val)); break;
      }
    }

    bool has_input_{};
    InputFormat input_{};
    bool has_output_{};
    OutputFormat output_{};
    bool has_network_{};
    NetworkStructure network_{};
    bool has_policy_{};
    PolicyFormat policy_{};
    bool has_value_{};
    ValueFormat value_{};
    bool has_moves_left_{};
    MovesLeftFormat moves_left_{};
  };
  class Format : public lczero::ProtoMessage {
   public:
    enum Encoding {
      UNKNOWN = 0,
      LINEAR16 = 1,
    };

    bool has_weights_encoding() const { return has_weights_encoding_; }
    Encoding weights_encoding() const { return weights_encoding_; }
    void set_weights_encoding(Encoding val) {
      has_weights_encoding_ = true;
      weights_encoding_ = val;
    }

    bool has_network_format() const { return has_network_format_; }
    const NetworkFormat& network_format() const { return network_format_; }
    NetworkFormat* mutable_network_format() {
      has_network_format_ = true;
      return &network_format_;
    }

    void Clear() override {
      has_weights_encoding_ = false;
      weights_encoding_ = {};
      has_network_format_ = false;
      network_format_ = {};
    }

   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_weights_encoding(static_cast<Encoding>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 2: mutable_network_format()->MergeFromString(val); break;
      }
    }

    bool has_weights_encoding_{};
    Encoding weights_encoding_{};
    bool has_network_format_{};
    NetworkFormat network_format_{};
  };
  class Net : public lczero::ProtoMessage {
   public:

    bool has_magic() const { return has_magic_; }
    std::uint32_t magic() const { return magic_; }
    void set_magic(std::uint32_t val) {
      has_magic_ = true;
      magic_ = val;
    }

    bool has_license() const { return has_license_; }
    std::string_view license() const { return license_; }
    void set_license(std::string_view val) {
      has_license_ = true;
      license_ = val;
    }

    bool has_min_version() const { return has_min_version_; }
    const EngineVersion& min_version() const { return min_version_; }
    EngineVersion* mutable_min_version() {
      has_min_version_ = true;
      return &min_version_;
    }

    bool has_format() const { return has_format_; }
    const Format& format() const { return format_; }
    Format* mutable_format() {
      has_format_ = true;
      return &format_;
    }

    bool has_training_params() const { return has_training_params_; }
    const TrainingParams& training_params() const { return training_params_; }
    TrainingParams* mutable_training_params() {
      has_training_params_ = true;
      return &training_params_;
    }

    bool has_weights() const { return has_weights_; }
    const Weights& weights() const { return weights_; }
    Weights* mutable_weights() {
      has_weights_ = true;
      return &weights_;
    }

    void Clear() override {
      has_magic_ = false;
      magic_ = {};
      has_license_ = false;
      license_ = {};
      has_min_version_ = false;
      min_version_ = {};
      has_format_ = false;
      format_ = {};
      has_training_params_ = false;
      training_params_ = {};
      has_weights_ = false;
      weights_ = {};
    }

   private:
    void SetInt32(int field_id, std::uint32_t val) override {
      switch (field_id) {
        case 1: set_magic(static_cast<std::uint32_t>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 2: set_license(val); break;
        case 3: mutable_min_version()->MergeFromString(val); break;
        case 4: mutable_format()->MergeFromString(val); break;
        case 5: mutable_training_params()->MergeFromString(val); break;
        case 10: mutable_weights()->MergeFromString(val); break;
      }
    }

    bool has_magic_{};
    std::uint32_t magic_{};
    bool has_license_{};
    std::string license_{};
    bool has_min_version_{};
    EngineVersion min_version_{};
    bool has_format_{};
    Format format_{};
    bool has_training_params_{};
    TrainingParams training_params_{};
    bool has_weights_{};
    Weights weights_{};
  };
}  // namespace pblczero
