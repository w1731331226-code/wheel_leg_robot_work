#!/usr/bin/env python3
"""
test_jacobian_improved.py

改进版测试程序，添加更多验证和调试信息
"""
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from jacobian import calculate_analytical_kinematics


def parse_user_j(s: str):
    """Parse user jacobian string like 'a,b,c,d' or JSON array string."""
    try:
        arr = json.loads(s)
        arr = list(arr)
    except Exception:
        arr = [float(x) for x in s.replace(',', ' ').split()]
    if len(arr) != 4:
        raise ValueError('user_J must have 4 elements (row-major)')
    return np.array([[arr[0], arr[1]], [arr[2], arr[3]]], dtype=float)


def load_cases_from_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def validate_jacobian(J):
    """检查雅可比矩阵的有效性"""
    issues = []
    
    # 检查 NaN/Inf
    if np.any(np.isnan(J)):
        issues.append("包含 NaN")
    if np.any(np.isinf(J)):
        issues.append("包含 Inf")
    
    # 检查行列式（奇异性）
    det = np.linalg.det(J)
    if np.abs(det) < 1e-8:
        issues.append(f"接近奇异 (det={det:.6g})")
    
    # 检查数值范围
    J_abs_max = np.max(np.abs(J))
    if J_abs_max > 1e6:
        issues.append(f"元素过大 (max={J_abs_max:.6g})")
    
    return issues


def run_case(case, tol=1e-6, verbose=True):
    params = case.get('params', {'L_OA':0.12, 'L_OB':0.2, 'L_AD':0.12, 'L_BC':0.2})
    thetaA = float(case['thetaA'])
    thetaB = float(case['thetaB'])

    user_J = None
    if 'user_J' in case and case['user_J'] is not None:
        user_J = np.array(case['user_J'], dtype=float)
        if user_J.shape == (4,):
            user_J = user_J.reshape((2,2))
        elif user_J.shape != (2,2):
            raise ValueError('user_J must be length-4 list or 2x2 list')

    # compute reference
    result = calculate_analytical_kinematics(thetaA, thetaB, params)
    if result[0] is None:
        if verbose:
            print(f"thetaA={thetaA}, thetaB={thetaB}: Invalid geometry (reference cannot be computed)")
        return False, None

    (Cx, Cy), J_ref = result

    # 验证参考雅可比
    ref_issues = validate_jacobian(J_ref)
    if ref_issues and verbose:
        print(f"⚠️  Reference J 有问题: {', '.join(ref_issues)}")

    if user_J is None:
        if verbose:
            print('No user_J provided; showing reference J:')
            print(J_ref)
        return True, J_ref

    # ensure shapes
    user_J = np.asarray(user_J, dtype=float)
    if user_J.shape != (2,2):
        raise ValueError('user_J must be 2x2')

    # 验证用户雅可比
    user_issues = validate_jacobian(user_J)
    if user_issues and verbose:
        print(f"⚠️  User J 有问题: {', '.join(user_issues)}")

    diff = np.abs(J_ref - user_J)
    max_err = float(np.max(diff))
    rel_err = max_err / (np.max(np.abs(J_ref)) + 1e-10)  # 相对误差

    ok = max_err <= tol
    if verbose:
        print(f'Case thetaA={thetaA} thetaB={thetaB}')
        print(f'Params: {params}')
        print('Reference J:')
        print(J_ref)
        print('User J:')
        print(user_J)
        print('Abs diff:')
        print(diff)
        print(f'max_abs_error = {max_err:.6g}')
        print(f'rel_error = {rel_err:.6g}')
        print(f'tol = {tol}')
        
        # 行列式对比
        det_ref = np.linalg.det(J_ref)
        det_user = np.linalg.det(user_J)
        print(f'Reference det: {det_ref:.6g}, User det: {det_user:.6g}')
        
        print('✅ PASS' if ok else '❌ FAIL')
        print('---')

    # 如果提供了 user_J，则做点级预测对比
    if user_J is not None:
        try:
            test_predictions(J_ref, user_J, thetaA, thetaB, Cx, Cy, params)
        except Exception as e:
            if verbose:
                print(f'Prediction test failed: {e}')

    return ok, max_err


def test_predictions(J_ref, user_J, thetaA, thetaB, Cx, Cy, params, verbose=True):
    """
    进行多个方向的小角增量预测对比
    """
    if verbose:
        print("📊 Point-level prediction tests:")
    
    dh_deg = 1e-3
    dh_rad = np.radians(dh_deg)
    
    test_dirs = [
        ("dthA only", np.array([dh_rad, 0.0])),
        ("dthB only", np.array([0.0, dh_rad])),
        ("diagonal", np.array([dh_rad, dh_rad]) / np.sqrt(2)),
    ]
    
    L0 = np.sqrt(Cx**2 + Cy**2)
    th0 = np.arctan2(Cy, Cx)
    
    max_pred_err = 0.0
    
    for name, dtheta in test_dirs:
        # 真实值：通过数值求解
        result_true = calculate_analytical_kinematics(
            thetaA + np.degrees(dtheta[0]),
            thetaB + np.degrees(dtheta[1]),
            params
        )
        
        if result_true[0] is None:
            if verbose:
                print(f"  {name}: Invalid geometry")
            continue
        
        (Cx_true, Cy_true), _ = result_true
        L_true = np.sqrt(Cx_true**2 + Cy_true**2)
        th_true = np.arctan2(Cy_true, Cx_true)
        
        # 用户预测：J_ref @ dtheta
        d_out_ref = J_ref @ dtheta
        L_pred_ref = L0 + float(d_out_ref[0])
        th_pred_ref = th0 + float(d_out_ref[1])
        
        # 用户预测：J_user @ dtheta
        d_out_user = user_J @ dtheta
        L_pred_user = L0 + float(d_out_user[0])
        th_pred_user = th0 + float(d_out_user[1])
        
        # 转换回笛卡尔坐标
        C_pred_ref = np.array([L_pred_ref * np.cos(th_pred_ref),
                                L_pred_ref * np.sin(th_pred_ref)])
        C_pred_user = np.array([L_pred_user * np.cos(th_pred_user),
                                 L_pred_user * np.sin(th_pred_user)])
        C_true = np.array([Cx_true, Cy_true])
        
        err_ref = np.linalg.norm(C_pred_ref - C_true)
        err_user = np.linalg.norm(C_pred_user - C_true)
        max_pred_err = max(max_pred_err, err_user)
        
        if verbose:
            print(f"  {name:15s}: ref_pred_err={err_ref:.6g}, user_pred_err={err_user:.6g}")
    
    if verbose:
        print(f"  Max prediction error: {max_pred_err:.6g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--thetaA', type=float, help='theta A in degrees')
    p.add_argument('--thetaB', type=float, help='theta B in degrees')
    p.add_argument('--params', type=str,
                   help='JSON string or file path for params dict')
    p.add_argument('--user-j', type=str, help="user jacobian as 'a,b,c,d' or JSON list")
    p.add_argument('--user-j-file', type=str, help='path to JSON file containing 4 numbers or 2x2 array')
    p.add_argument('--cases', type=str, help='path to JSON file with list of test cases')
    p.add_argument('--tol', type=float, default=1e-6, help='absolute tolerance')
    args = p.parse_args()

    if args.cases:
        cases = load_cases_from_file(args.cases)
        all_ok = True
        worst = 0.0
        for i, c in enumerate(cases):
            print(f"\n=== Case {i+1}/{len(cases)} ===")
            ok, val = run_case(c, tol=args.tol, verbose=True)
            all_ok = all_ok and ok
            if val is not None and val > worst:
                worst = val
        if not all_ok:
            print(f'\n❌ One or more cases failed. worst_error={worst:.6g}')
            sys.exit(2)
        else:
            print(f'\n✅ All cases passed. worst_error={worst:.6g}')
            sys.exit(0)

    if args.thetaA is None or args.thetaB is None:
        p.print_help()
        sys.exit(1)

    case = {'thetaA': args.thetaA, 'thetaB': args.thetaB}
    if args.params:
        try:
            params = json.loads(args.params)
        except Exception:
            with open(args.params, 'r') as f:
                params = json.load(f)
        case['params'] = params

    if args.user_j:
        case['user_J'] = list(parse_user_j(args.user_j).reshape(-1))
    elif args.user_j_file:
        with open(args.user_j_file, 'r') as f:
            uj = json.load(f)
        case['user_J'] = uj

    ok, val = run_case(case, tol=args.tol, verbose=True)
    if not ok:
        sys.exit(2)
    sys.exit(0)


if __name__ == '__main__':
    main()
