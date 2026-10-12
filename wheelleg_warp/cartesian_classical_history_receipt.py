"""Independent first-episode H1 receipt from actual deliveries and dense controls."""
import numpy as np
from review_cartesian_light_history import Receipt, fresh

PRIVATE = ('guard_memory', 'parking_seen', 'parking_effective', 'native_controller',
           'phase_reference', 'cart_latent', 'cart_shadow', 'cart_reference')
DELIVERY = ('prewait_public38', 'prewait_count', 'previous_sensor_steps_before', 'delay_steps',
            'command_total_before', 'command_prefix_before', 'command_stamp_before',
            'interval_integral', 'interval_elapsed_s', 'terminal_worlds',
            'command_total_after', 'command_prefix_after', 'command_stamp_after',
            'post_previous_sensor_steps', 'postwait_public38', 'returned_raw481')


def pairs(a):
    return np.stack((a, -a), axis=-1).reshape(*a.shape[:-1], 6)


def equal(a, b):
    np.testing.assert_array_equal(a, b)


def encoded(snapshot):
    return np.concatenate((snapshot['frames'].reshape(390), snapshot['inputs'].reshape(72),
                           snapshot['elapsed'], snapshot['valid']))


def public_reference(packet):
    from state_estimation import leg_kinematics
    return np.concatenate([leg_kinematics(packet[12+2*s:14+2*s].astype(float), np.zeros(2))[0]
                           for s in (0, 1)])


def review_case(*, case, row, world, initial, actor, full, cart, parking, guard,
                actual_terminal, after_reset, ids, delivery):
    """Inputs are loaded arrays; caller separately binds hashes, model and source scope."""
    if delivery is None:
        raise ValueError('Missing actual per-packet delivery/private evidence')
    required = set(DELIVERY) | {n + suffix for n in PRIVATE for suffix in ('_before', '_after')}
    missing = required - delivery.keys()
    if missing:
        raise ValueError('Missing actual delivery fields: ' + ','.join(sorted(missing)))
    if type(world) is not int or not 0 <= world < len(initial['public']):
        raise ValueError('Explicit valid batch world required')
    if row['seed'] != case['seed'] or row['scenario'] != case['scenario']:
        raise ValueError('History belongs to a different case')
    n = row['physical_steps']; packets = (n + 39) // 40
    assert type(n) is int and n > 0 and row['duration_s'] == n * .0005
    actor = np.asarray(actor); g = np.asarray(guard); park = np.asarray(parking); c = cart['trace']
    assert actor.shape == (packets, 968) and actor.dtype == np.float32
    assert g.shape == (n, 62) and park.shape == (n, 30) and c.shape == (n, 38)
    assert full['pre'].shape == full['post'].shape == (n, 39)
    assert full['trace'].shape == (n, 112) and cart['clock'].shape == (n, 3)
    equal(cart['clock'], full['trace'][:, 1:4])
    equal(cart['clock'][:, 0], np.arange(1, n + 1))
    assert all(np.isfinite(v).all() for v in (actor, g, park, c, full['pre'], full['post'], full['trace'], cart['clock'], actual_terminal))
    assert all(len(delivery[k]) == packets and np.isfinite(delivery[k]).all() for k in required)
    equal(actor[:, :481], actor[:, 481:962])
    equal(g[:, 0], 1); equal(g[:, 1], np.arange(1, n + 1)); equal(c[:, 0], 1)
    equal(park[:, 0], 1); equal(park[:, 1], np.arange(1, n + 1))
    equal(g[:, 20:26], full['pre'][:, 33:39])
    equal(c[0, 1:4], 0); equal(c[1:, 1:4], c[:-1, 4:7])
    equal(c[:, 4:7], c[:, 1:4] + np.clip(park[:, 11:17:2] - c[:, 1:4], -.01, .01))
    equal(park[:, 17:23], pairs(c[:, 1:4])); equal(park[:, 23:29], pairs(c[:, 4:7]))
    native_ids = np.asarray(ids)
    assert native_ids.ndim == 1 and len(native_ids) >= 19 and np.issubdtype(native_ids.dtype, np.integer)
    qids, vids = native_ids[:4], native_ids[4:10]
    assert np.all((qids >= 0) & (qids < 17)) and np.all((vids >= 0) & (vids < 16))
    delay = initial['param'][world, 4]
    assert delay == int(delay) and 0 <= delay <= 20
    assert delay == round(case['scenario']['delay_ms'] * 2)
    receipt = Receipt(initial['public'][world:world + 1])
    equal(initial['cart_latent'][world], 0); equal(initial['cart_shadow'][world, 16:22], 0)
    equal(initial['public'][world, 12:16], initial['q'][world, qids].astype(np.float32))
    np.testing.assert_allclose(initial['cart_reference'][world], public_reference(initial['public'][world]), rtol=0, atol=1e-7)
    equal(initial['raw481'][world], receipt.actor[0])
    equal(actor[0, :481], receipt.actor[0])
    max_integral_error = 0.
    for k in range(packets):
        start, end = 40 * k, min(40 * (k + 1), n)
        item = {name: delivery[name][k] for name in required}
        done = k == packets - 1
        assert item['terminal_worlds'].dtype == np.bool_ and bool(item['terminal_worlds']) == done
        equal(item['prewait_count'], end); equal(item['delay_steps'], delay)
        equal(actor[k, 383:389], pairs(c[start:start + 1, 1:4])[0].astype(np.float32))
        equal(park[start:end, 5:11], np.broadcast_to(actor[k, 962:968], (end - start, 6)))
        equal(item['parking_effective_before'], actor[k, 962:968])
        equal(item['prewait_public38'][32:38], pairs(c[end - 1:end, 4:7])[0].astype(np.float32))
        equal(item['cart_latent_before'], c[end - 1, 4:7])
        equal(item['cart_shadow_before'][16:22], pairs(c[end - 1:end, 4:7])[0])
        equal(item['native_controller_before'][16:22], c[end - 1, 7:13])
        equal(item['cart_reference_before'], initial['cart_reference'][world])
        # Check delayed encoders/rotors directly against the bracketed physical stream.
        sensor_end = max(end - int(delay), 0)
        expected_qv = np.r_[full['post'][sensor_end - 1, qids],
                            full['post'][sensor_end - 1, 17 + vids]] if sensor_end else initial['public'][world, 12:22]
        equal(item['prewait_public38'][12:22], expected_qv.astype(np.float32))
        saved_g = np.zeros((40, 1, 62)); saved_g[:end - start, 0] = g[start:end]
        state = np.zeros((1, 39)); state[0, 0] = end
        r = dict(actor=actor[k:k + 1, :481], returned=item['returned_raw481'][None, :],
            state=state, done=np.array([done]), guard=saved_g,
            delay_steps=np.array([delay]), previous_sensor_steps=np.array([item['previous_sensor_steps_before']]),
            command_total=item['command_total_before'][None, :],
            command_prefix=item['command_prefix_before'][None, :, :],
            command_stamp=item['command_stamp_before'][None, :],
            command_integral=item['interval_integral'][None, :], sensor_elapsed_s=np.array([item['interval_elapsed_s']]),
            terminal_packet=item['prewait_public38'][None, :], public_after=item['postwait_public38'][None, :])
        if done:
            r['terminal_history'] = np.asarray(actual_terminal)[None, :]
        receipt.consume(r)
        max_integral_error = max(max_integral_error, receipt.max_integral_error)
        for name in PRIVATE:
            if not done or name in ('parking_effective', 'phase_reference'):
                equal(item[name + '_after'], item[name + '_before'])
        equal(item['command_stamp_after'], receipt.stamp[0])
        np.testing.assert_allclose(item['command_total_after'], receipt.total[0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(item['command_prefix_after'], receipt.prefix[0], rtol=0, atol=1e-10)
        equal(item['post_previous_sensor_steps'], receipt.previous[0])
    terminal_buffers = cart['terminal']
    for name in ('latent', 'shadow', 'reference'):
        equal(terminal_buffers[name], item['cart_' + name + '_before'])
    equal(terminal_buffers['canonical'], item['native_controller_before'][16:22])
    for name, key in (('guard_memory', 'guard_memory'), ('parking_seen', 'parking_seen'),
                      ('parking_effective', 'parking_effective'), ('native_controller', 'controller_memory'),
                      ('phase_reference', 'phase_reference'), ('cart_latent', 'cart_latent'),
                      ('cart_shadow', 'cart_shadow'), ('cart_reference', 'cart_reference')):
        equal(item[name + '_after'], after_reset[key])
    equal(item['postwait_public38'], after_reset['public'])
    equal(item['returned_raw481'], encoded(after_reset))
    equal(item['returned_raw481'], fresh(after_reset['public'][None, :])[0])
    equal(after_reset['public'], initial['public'][world])
    equal(after_reset['param'], initial['param'][world])
    equal(after_reset['q'], initial['q'][world]); equal(after_reset['v'], 0)
    equal(after_reset['stopped_q'], full['post'][-1, :17]); equal(after_reset['stopped_v'], full['post'][-1, 17:33])
    equal(after_reset['ctrl'], full['pre'][-1, 33:39])
    equal(after_reset['guard_memory'], 0); equal(after_reset['parking_seen'], 0)
    equal(after_reset['cart_latent'], 0); equal(after_reset['cart_shadow'][16:22], 0)
    equal(after_reset['cart_shadow'][:16], item['cart_shadow_before'][:16])
    equal(after_reset['total'], 0); equal(after_reset['prefix'], 0)
    stamp = np.full(82, -1, np.int32); stamp[0] = 0
    equal(after_reset['stamp'], stamp); equal(after_reset['previous_sensor_steps'], 0)
    equal(after_reset['active'], 1); equal(after_reset['done'], 0); equal(after_reset['targets'], 0)
    expected_state = np.zeros(39); expected_state[[1, 30, 31, 32, 33, 38]] = [-1, 1, 1, 1, 1e30, 1e30]
    equal(after_reset['state'], expected_state)
    native = np.zeros(30); native[1] = after_reset['native_reference'][2]; native[12] = -1
    equal(after_reset['controller_memory'], native)
    equal(after_reset['native_reference'], initial['native_reference'][world])
    np.testing.assert_allclose(after_reset['cart_reference'], public_reference(after_reset['public']), rtol=0, atol=1e-7)
    return dict(verified=True, steps=n, history_rows=packets, actual_terminal_and_all_returned481_verified=True,
        command_integrals_and_delayed_encoder_windows_verified=True, max_command_integral_error=max_integral_error,
        same_packet_named_private_and_collector_protection_verified=True, first_episode_reset_endpoints_verified=True,
        full_body_sensor_origin_independently_reconstructed=False,
        phase_role_monitor_full_private_protection_independently_verified=False,
        scope='Actual public38 H1/route, dense final commands, delayed q/v, listed private buffers and first-episode reset; no curriculum or full sensor reconstruction.')
