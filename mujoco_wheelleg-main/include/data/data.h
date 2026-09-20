#pragma once
#include <mujoco/mujoco.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Sensor implementation: reads sensordata by sensor name
class Sensor {
 public:
    explicit Sensor(const char* name) : name_(name ? name : "") {}

    void getData(const mjModel* m, const mjData* d, double* sensor_data){
        int idx = mj_name2id(m, mjOBJ_SENSOR, name_.c_str());
        if (idx < 0) { return; }
        int adr = m->sensor_adr[idx];
        int dim = m->sensor_dim[idx];
        for (int i = 0; i < dim; ++i) sensor_data[i] = d->sensordata[adr + i];
    }

    int getDataDim(const mjModel* m) {
        int idx = mj_name2id(m, mjOBJ_SENSOR, name_.c_str());
        if (idx < 0) return 0;
        return m->sensor_dim[idx];
    }

    const char* getDataName() { return name_.c_str(); }

    void setData(mjModel* m, mjData* d, const double* values) {
        printf("You can't assign a value to a sensor! Action aborted.\n");
    }

    void printData(const mjModel* m, const mjData* d) {
        int idx = mj_name2id(m, mjOBJ_SENSOR, name_.c_str());
        if (idx < 0) {
            printf("%s: \n", name_.c_str());
            return;
        }
        int adr = m->sensor_adr[idx];
        int dim = m->sensor_dim[idx];
        printf("%s:", name_.c_str());
        for (int i = 0; i < dim; ++i) printf(" %g", d->sensordata[adr + i]);
        printf("\n");
    }

 private:
    std::string name_;
};

// Joint implementation: read/write joint qpos values by joint name.
// getData returns the joint's qpos entries (dimension may be >1 for e.g. ball joints).
class Joint {
 public:
    explicit Joint(const char* name) : name_(name ? name : "") {}

    // Read qpos values for this joint into out (length = getDataDim(m)).
    void getData(const mjModel* m, const mjData* d, double* out) {
        int jid = mj_name2id(m, mjOBJ_JOINT, name_.c_str());
        if (jid < 0) return;
        int qposadr = m->jnt_qposadr[jid];
        int qposdim = Joint::qposDim(m, jid);
        for (int i = 0; i < qposdim; ++i) out[i] = d->qpos[qposadr + i];
    }

    int getDataDim(const mjModel* m) {
        int jid = mj_name2id(m, mjOBJ_JOINT, name_.c_str());
        if (jid < 0) return 0;
        return Joint::qposDim(m, jid);
    }

    const char* getDataName() { return name_.c_str(); }

    void setData(mjModel* m, mjData* d, const double* values) {
        printf("Note that assigning a value to a joint is disregarding the laws of the physics simulation. Action aborted.\n");
    }

    void printData(const mjModel* m, const mjData* d) {
        int jid = mj_name2id(m, mjOBJ_JOINT, name_.c_str());
        if (jid < 0) {
            printf("%s: \n", name_.c_str());
            return;
        }
        int qposadr = m->jnt_qposadr[jid];
        int qposdim = Joint::qposDim(m, jid);
        printf("%s:", name_.c_str());
        for (int i = 0; i < qposdim; ++i) printf(" %g", d->qpos[qposadr + i]);
        printf("\n");
    }

 private:
    static int qposDim(const mjModel* m, int jid) {
        if (jid < 0 || !m) return 0;
        if (jid + 1 < m->njnt) return m->jnt_qposadr[jid + 1] - m->jnt_qposadr[jid];
        return m->nq - m->jnt_qposadr[jid];
    }

    std::string name_;
};

// Actuator implementation: read current control input and current applied actuator value
// getData returns two values: [ctrl, act] where
//  - ctrl is the commanded control value from d->ctrl (if available)
//  - act is the currently applied actuator value from d->act (if available)
// setData writes the control value into d->ctrl (values[0]).
class Actuator {
 public:
    explicit Actuator(const char* name) : name_(name ? name : "") {}

    void getData(const mjModel* m, const mjData* d, double* out) {
        int aid = mj_name2id(m, mjOBJ_ACTUATOR, name_.c_str());
        if (aid < 0) {
            out[0] = out[1] = 0.0;
            return;
        }
        // ctrl index: only valid if within m->nu
        double ctrl = 0.0;
        if (aid >= 0 && aid < m->nu) ctrl = d->ctrl[aid];
        double act = 0.0;
        if (aid >= 0 && aid < m->na) act = d->act[aid];
        out[0] = ctrl;
        out[1] = act;
    }

    int getDataDim(const mjModel* /*m*/) { return 2; }

    const char* getDataName() { return name_.c_str(); }

    void setData(mjModel* m, mjData* d, const double* values) {
        int aid = mj_name2id(m, mjOBJ_ACTUATOR, name_.c_str());
        if (aid < 0) return;
        if (aid >= 0 && aid < m->nu) d->ctrl[aid] = values[0];
        // Note: d->act is typically computed by MuJoCo; do not write it here.
    }

    void printData(const mjModel* m, const mjData* d) {
        int aid = mj_name2id(m, mjOBJ_ACTUATOR, name_.c_str());
        printf("%s:", name_.c_str());
        if (aid < 0) {
            printf("\n");
            return;
        }
        double ctrl = 0.0;
        double act = 0.0;
        if (aid >= 0 && aid < m->nu) ctrl = d->ctrl[aid];
        if (aid >= 0 && aid < m->na) act = d->act[aid];
        printf(" %g %g\n", ctrl, act);
    }

 private:
    std::string name_;
};

// Build a cached list of IData objects for the given sensor names.
// After calling BuildSensors you can call PrintCachedSensors repeatedly
// without reconstructing the IData objects.
void BuildSensors(const mjModel* m, const char** names, int n);
// Print the previously built cached sensors. No-op if none built.
void PrintCachedSensors(const mjModel* m, const mjData* d);
// Clear any cached sensor objects.
void ClearSensors();

// Joint cached API: build, print, clear (prints joint qpos/qvel like sensors/actuators)
void BuildJoints(const mjModel* m, const char** names, int n);
void PrintCachedJoints(const mjModel* m, const mjData* d);
void ClearJoints();

// Print both sensors and joints together, rate-limited to at most once per second.
void PrintCachedGroups(const mjModel* m, const mjData* d);

// Build cached actuator objects for the given actuator names.
void BuildActuators(const mjModel* m, const char** names, int n);
// Set actuators from an array of values (one value per cached actuator).
// The array length should be >= number of cached actuators; extra values ignored.
void SetCachedActuators(const mjModel* m, mjData* d, const double* values, int n);
// Clear any cached actuator objects.
void ClearActuators();

#ifdef __cplusplus
}
#endif

// g_cached_* 是 C++ 对象，保持 C++ 链接，定义在 extern "C" 块之外。
inline std::vector<std::unique_ptr<Sensor>> g_cached_sensors;
inline std::vector<std::unique_ptr<Joint>> g_cached_joints;
inline std::vector<std::unique_ptr<Actuator>> g_cached_actuators;