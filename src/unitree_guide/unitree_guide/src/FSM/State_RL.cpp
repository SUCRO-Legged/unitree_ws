/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/State_RL.h"

State_RL::State_RL(CtrlComponents *ctrlComp) : FSMState(ctrlComp, FSMStateName::RL, "rl"), actorSession(nullptr), encoderSession(nullptr)
{
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "            Reinforcement Learning Warm Up and Init           " << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    std::string actorPath = "actor.onnx";
    std::string encoderPath = "encoder.onnx";
    actorPath = "/model/" + actorPath;
    encoderPath = "/model/" + encoderPath;
    actorPath = std::string(get_current_dir_name()) + actorPath;
    encoderPath = std::string(get_current_dir_name()) + encoderPath;
    std::cout << "load actor model from " << actorPath << std::endl;
    std::cout << "load encoder model from " << encoderPath << std::endl;

    // init ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
    // Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "ONNXRuntimeInference");

    // create Session Options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // load ONNX model
    encoderSession = Ort::Session(env, encoderPath.c_str(), session_options);
    actorSession = Ort::Session(env, actorPath.c_str(), session_options);

    if (!actorSession || !encoderSession)
        std::cerr << "Failed to create session!" << std::endl;
    else
        std::cout << "session created ..." << std::endl;

    reset();

    for (int i = 0; i < 10; i++)
    {
        std::chrono::steady_clock::time_point _end = std::chrono::steady_clock::now();
        infer();
        std::chrono::steady_clock::time_point _begin = std::chrono::steady_clock::now();
        if (i == 0)
            std::cout << "[INIT]: " << "The policy first forward time duration = " << std::chrono::duration_cast<std::chrono::microseconds>(_begin - _end).count() << " us" << std::endl;
        if (i >= 9)
            std::cout << "[INIT]: " << "The policy last forward time duration = " << std::chrono::duration_cast<std::chrono::microseconds>(_begin - _end).count() << " us" << std::endl;
    }

    std::cout << "[INFO]: State_RL is ok!" << std::endl;
}

void State_RL::enter()
{
    std::cout << "[INFO]: Start Joint Position Control learned by RL" << std::endl;
    std::cout << "[INFO]: RL FSM Start Init!" << std::endl;

    // unitree_guide: FR FL RR RL
    // _defaultDofPos << -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1., -1.5, 0.1, 1., -1.5;
    _defaultDofPos << -0.0, 0.8, -1.5, 0.0, 0.8, -1.5, -0.0, 0.8, -1.5, 0.0, 0.8, -1.5;

    for (int i = 0; i < 12; i++)
        _targetDofPos[i] = _defaultDofPos[i];

    // init command
    _command.setZero();
    _vCmdBody.setZero();
    _dYawCmd = 0.0;
    reset();

    std::cout << "[INFO]: RL FSM RUN Start!" << std::endl;
}

void State_RL::reset()
{
    // reset observations
    memset(obsPtr, 0, sizeof(float) * NUM_OBS);
    // reset latent representation
    memset(latentPtr, 0, sizeof(float) * NUM_LATENT);
    // reset hidden and cell state of encoder and actor
    memset(hePtr, 0, sizeof(float) * HIDDEN_SIZE);
    memset(cePtr, 0, sizeof(float) * HIDDEN_SIZE);
    memset(haPtr, 0, sizeof(float) * HIDDEN_SIZE);
    memset(caPtr, 0, sizeof(float) * HIDDEN_SIZE);
    // reset actions
    memset(actionsPtr, 0, sizeof(float) * NUM_ACTIONS);

    this->count = this->decimation;
    std::cout << "[INFO]: Reset Hidden State and Cell State!" << std::endl;
}

void State_RL::run() // 500Hz
{
    ++this->count;
    if (this->count > this->decimation) // 100Hz
    {
        computeObservations();
        infer();
        this->count = 0;
    }
    step();
}

void State_RL::infer()
{
    encoder_input_tensors.clear();
    encoder_output_tensors.clear();
    actor_input_tensors.clear();
    actor_output_tensors.clear();

    encoder_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        obsPtr, NUM_OBS,
        encoderSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().data(),
        encoderSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size()));
    encoder_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        hePtr, HIDDEN_SIZE,
        encoderSession.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape().data(),
        encoderSession.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape().size()));
    encoder_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        cePtr, HIDDEN_SIZE,
        encoderSession.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape().data(),
        encoderSession.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape().size()));

    encoder_output_tensors = encoderSession.Run(Ort::RunOptions{nullptr},
                                                encoder_input_node_names.data(),
                                                encoder_input_tensors.data(),
                                                encoder_input_tensors.size(),
                                                encoder_output_node_names.data(),
                                                encoder_output_node_names.size());

    latentPtr = encoder_output_tensors[0].GetTensorMutableData<float>();
    hePtr = encoder_output_tensors[1].GetTensorMutableData<float>();
    cePtr = encoder_output_tensors[2].GetTensorMutableData<float>();

    actor_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        obsPtr, NUM_OBS,
        actorSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().data(),
        actorSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size()));
    actor_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        latentPtr, NUM_LATENT,
        actorSession.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape().data(),
        actorSession.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape().size()));
    actor_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        haPtr, HIDDEN_SIZE,
        actorSession.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape().data(),
        actorSession.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape().size()));
    actor_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        caPtr, HIDDEN_SIZE,
        actorSession.GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape().data(),
        actorSession.GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape().size()));

    actor_output_tensors = actorSession.Run(
        Ort::RunOptions{nullptr},       // run options
        actor_input_node_names.data(),  // input names
        actor_input_tensors.data(),     // input values
        actor_input_tensors.size(),     // input values count
        actor_output_node_names.data(), // output names
        actor_output_node_names.size()  // output names count
    );

    actionsPtr = actor_output_tensors[0].GetTensorMutableData<float>();
    haPtr = actor_output_tensors[1].GetTensorMutableData<float>();
    caPtr = actor_output_tensors[2].GetTensorMutableData<float>();

    for (int i = 0; i < NUM_ACTIONS; i++)
    {
        _actions[i] = actionsPtr[i];
        _targetDofPos[i] = _actions[srDofIndices[i]] * this->actionScale + _defaultDofPos[i];
    }
}

void State_RL::step()
{
    for (int j = 0; j < 12; j++)
    {
        _lowCmd->motorCmd[j].mode = 10;
        _lowCmd->motorCmd[j].q = this->_targetDofPos[j];
        _lowCmd->motorCmd[j].dq = 0;
        _lowCmd->motorCmd[j].Kp = this->Kp;
        _lowCmd->motorCmd[j].Kd = this->Kd;
        _lowCmd->motorCmd[j].tau = 0;
    }
}

void State_RL::computeObservations()
{
    _B2G_RotMat = _lowState->getRotMat();
    _G2B_RotMat = _B2G_RotMat.transpose();
    _projGravity = _G2B_RotMat * _G;

    getUserCmd();

    // concatenate observations
    for (size_t i = 0; i < 3; i++)
    {
        obsPtr[0 + i] = _lowState->imu.gyroscope[i];
        obsPtr[3 + i] = _projGravity[i];
        obsPtr[6 + i] = _command[i];
    }
    for (int i = 0; i < 12; i++)
    {
        obsPtr[9 + i] = (_lowState->motorState[rlDofIndices[i]].q - _defaultDofPos[i]);
        obsPtr[9 + NUM_ACTIONS + i] = _lowState->motorState[rlDofIndices[i]].dq; // get dof vel
        obsPtr[9 + 2 * NUM_ACTIONS + i] = _actions[i];                           // get last actions
    }
}

void State_RL::getUserCmd()
{
    _userValue = _lowState->userValue;

    /* Movement */
    _vCmdBody(0) = invNormalize(_userValue.ly, _vxLim(0), _vxLim(1));
    _vCmdBody(1) = -invNormalize(_userValue.lx, _vyLim(0), _vyLim(1));

    /* Turning */
    _dYawCmd = -invNormalize(_userValue.rx, _wyawLim(0), _wyawLim(1));
    _dYawCmd = 0.9 * _dYawCmdPast + (1.0 - 0.9) * _dYawCmd;
    _dYawCmdPast = _dYawCmd;

    _command << _vCmdBody(0), _vCmdBody(1), _dYawCmd;
}

FSMStateName State_RL::checkChange()
{
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    else if (_lowState->userCmd == UserCommand::L2_A)
    {
        return FSMStateName::FIXEDSTAND;
    }
    else
    {
        return FSMStateName::RL;
    }
}

void State_RL::exit()
{
    this->count = 0;
}

State_RL::~State_RL()
{
}