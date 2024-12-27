/**********************************************************************
 RL simulation and deploy
***********************************************************************/
#ifndef RL_H
#define RL_H

#include "FSMState.h"
#include "common/mathTypes.h"
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <atomic>

#define NUM_LEGS 4
#define NUM_ACTIONS NUM_LEGS * 3
#define NUM_OBS 9 + NUM_ACTIONS * 3
#define NUM_LATENT 27
#define HIDDEN_SIZE 256

class State_RL : public FSMState
{
private:
    Ort::Session encoderSession;
    Ort::Session actorSession;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    float *obsPtr = new float[NUM_OBS];
    float *latentPtr = new float[NUM_LATENT];
    float *hePtr = new float[HIDDEN_SIZE];
    float *cePtr = new float[HIDDEN_SIZE];
    float *haPtr = new float[HIDDEN_SIZE];
    float *caPtr = new float[HIDDEN_SIZE];
    float *actionsPtr = new float[NUM_ACTIONS];

    std::vector<const char *> encoder_input_node_names = {"obs", "h_in", "c_in"};
    std::vector<const char *> encoder_output_node_names = {"latent", "h_out", "c_out"};
    std::vector<const char *> actor_input_node_names = {"obs", "latent", "h_in", "c_in"};
    std::vector<const char *> actor_output_node_names = {"actions", "h_out", "c_out"};

    std::vector<Ort::Value> actor_input_tensors, actor_output_tensors;
    std::vector<Ort::Value> encoder_input_tensors, encoder_output_tensors;
    std::chrono::steady_clock::time_point begin_;
    std::chrono::steady_clock::time_point end_;

    double _yaw, _dYaw;
    Vec3 _velBody, _angleBody;
    RotMat _B2G_RotMat, _G2B_RotMat;
    Vec3 _projGravity;

    Eigen::Matrix<float, NUM_ACTIONS, 1> _targetDofPos, _last_targetDofPos, _defaultDofPos, _actions, _dofPos, _dofVel;

    // robot command
    Vec3 _vCmdBody;
    double _dYawCmd, _dYawCmdPast;
    Vec3 _command;

    // issacgym
    // FR_hip FR_thigh FR_calf
    const int srDofIndices[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8}; // To Gazebo and Real Robot
    // FL_hip FL_thigh FL_calf
    const int rlDofIndices[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8}; // To RL Training

    // // issaclab
    // // FR_hip FR_thigh FR_calf
    // const int srDofIndices[12] = {1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10}; // To Gazebo and Real Robot
    // // FL_hip FR_hip
    // const int rlDofIndices[12] = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8}; // To RL Training

    // // genesis
    // // FR_hip FR_thigh FR_calf
    // const int srDofIndices[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // To Gazebo and Real Robot
    // // FL_hip FL_thigh FL_calf
    // const int rlDofIndices[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // To RL Training

    const Vec3 _G = {0, 0, -1.0};
    const Vec2 _vxLim = {-1.0, 1.0}, _vyLim = {-1.0, 1.0}, _wyawLim = {-1.57, 1.57};

    // scales
    const double actionScale = 0.25;
    const double Kp = 20.0; // stiffness
    const double Kd = 0.5;  // damping

    const double dt = 0.01;      // RL control dt 0.01
    const double decimation = 5; // RL control freqency 50hz
    size_t count = 0;

    void reset();
    void infer();
    void step();
    void computeObservations();
    virtual void getUserCmd();

public:
    State_RL(CtrlComponents *ctrlComp);
    ~State_RL();
    void enter();
    void run();
    void exit();
    virtual FSMStateName checkChange();
};
#endif