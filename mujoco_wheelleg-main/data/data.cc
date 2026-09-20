#include <mujoco/mujoco.h>
#include "data.h"
#include "get_data.h"
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>
#include <mutex>
#include "alg_vmc.h"

static bool g_sensors_built = false;

extern "C" void BuildSensors(const mjModel* /*m*/, const char** names, int n) {
    // Build only once per program run unless ClearSensors() is called.
    if (g_sensors_built) return;
    g_cached_sensors.clear();
    if (!names || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        const char* nm = names[i];
        if (!nm) continue;
        g_cached_sensors.emplace_back(new Sensor(nm));
    }
    g_sensors_built = true;
}

extern "C" void PrintCachedSensors(const mjModel* m, const mjData* d) {
    for (auto& p : g_cached_sensors) {
        if (p) p->printData(m, d);
    }
    // extra blank line after printing the whole sensor group
    printf("\n");
}

extern "C" void ClearSensors() {
    g_cached_sensors.clear();
    g_sensors_built = false;
}

static bool g_joints_built = false;

extern "C" void BuildJoints(const mjModel* /*m*/, const char** names, int n) {
    if (g_joints_built) return;
    g_cached_joints.clear();
    if (!names || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        const char* nm = names[i];
        if (!nm) continue;
        g_cached_joints.emplace_back(new Joint(nm));
    }
    g_joints_built = true;
}

extern "C" void PrintCachedJoints(const mjModel* m, const mjData* d) {
    if (!g_joints_built) return;
    for (auto& p : g_cached_joints) {
        if (p) p->printData(m, d);
    }
    // extra blank line after printing the whole joint group
    printf("\n");
}

extern "C" void ClearJoints() {
    g_cached_joints.clear();
    g_joints_built = false;
}

static bool g_actuators_built = false;

extern "C" void BuildActuators(const mjModel* /*m*/, const char** names, int n) {
    if (g_actuators_built) return;
    g_cached_actuators.clear();
    if (!names || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        const char* nm = names[i];
        if (!nm) continue;
        g_cached_actuators.emplace_back(new Actuator(nm));
    }
    g_actuators_built = true;
}

// Set cached actuators: values is an array with one value per cached actuator
extern "C" void SetCachedActuators(const mjModel* m, mjData* d, const double* values, int n) {
    if (!g_actuators_built || !values || n <= 0) return;
    int count = (int)g_cached_actuators.size();
    int lim = n < count ? n : count;
    for (int i = 0; i < lim; ++i) {
        if (g_cached_actuators[i]) {
            // setData expects pointer to values for that actuator (we use first element)
            g_cached_actuators[i]->setData(const_cast<mjModel*>(m), d, &values[i]);
        }
    }
}

extern "C" void ClearActuators() {
    g_cached_actuators.clear();
    g_actuators_built = false;
}

// Combined print for sensors+joints, rate-limited to at most once per second.
extern "C" void PrintCachedGroups(const mjModel* m, const mjData* d) {
    static std::mutex g_print_mutex;
    using Clock = std::chrono::steady_clock;
    static Clock::time_point g_last_print_time = Clock::now() - std::chrono::seconds(2);

    std::lock_guard<std::mutex> lock(g_print_mutex);
    auto now = Clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_print_time).count();
    if (elapsed < 1000) return; // less than 1s since last print
    g_last_print_time = now;

    // populate inline variables from cached sensors/joints
    UpdateInlineGetData(m, d);

        // read wheel ground force sensors (each may be 3D: fx,fy,fz)
        double w1[3] = {0.0, 0.0, 0.0};
        double w2[3] = {0.0, 0.0, 0.0};
        int id1 = mj_name2id(m, mjOBJ_SENSOR, "wheel1_ground_force");
        if (id1 >= 0) {
            int adr = m->sensor_adr[id1];
            int dim = m->sensor_dim[id1];
            for (int i = 0; i < mjMIN(dim, 3); ++i) w1[i] = d->sensordata[adr + i];
        }
        int id2 = mj_name2id(m, mjOBJ_SENSOR, "wheel2_ground_force");
        if (id2 >= 0) {
            int adr = m->sensor_adr[id2];
            int dim = m->sensor_dim[id2];
            for (int i = 0; i < mjMIN(dim, 3); ++i) w2[i] = d->sensordata[adr + i];
        }
        // w1/w2 目前仅被上方注释掉的 printf 使用，显式标记避免 -Wunused-but-set-variable
        (void)w1;
        (void)w2;

    // print the inline values (one line per logical group)
    // printf("body_euler pitch roll yaw: %g %g %g\n", (double)body_pitch, (double)body_roll, (double)body_yaw);
    // printf("body_accel x y z: %g %g %g\n", (double)body_ax, (double)body_ay, (double)body_az);
    // printf("body_gyro pitch roll yaw: %g %g %g\n", (double)body_gpitch, (double)body_groll, (double)body_gyaw);
    // printf("wheel_speed1 wheel_speed2: %g %g\n", (double)wheel_speed1, (double)wheel_speed2);
    // printf("thigh1 angle vel: %g %g\n", (double)thigh1_angle, (double)thigh1_vel);
    // printf("rod1 angle vel: %g %g\n", (double)rod1_angle, (double)rod1_vel);
    // printf("thigh2 angle vel: %g %g\n", (double)thigh2_angle, (double)thigh2_vel);
    // printf("rod2 angle vel: %g %g\n", (double)rod2_angle, (double)rod2_vel);
    // printf("wheel1 ground force (fx,fy,fz): %g %g %g\n", w1[0], w1[1], w1[2]);
    // printf("wheel2 ground force (fx,fy,fz): %g %g %g\n", w2[0], w2[1], w2[2]);
    // printf("leg1_length leg2_length: %g %g\n", (double)leg1_length, (double)leg2_length);
    // printf("left_sum left_diff right_sum right_diff: %g %g %g %g\n", (double)left_sum, (double)left_diff, (double)right_sum, (double)right_diff);
    // single blank line after both groups
    // printf("\n");
    printf("leftwheelpos rightwheelpos: %g %g\n", (double)wheel_angle1, (double)wheel_angle2);
}