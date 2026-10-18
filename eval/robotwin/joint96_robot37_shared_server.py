"""One immutable runtime per GPU, isolated sessions, serialized inference."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import numpy as np
from eval.robotwin.joint96_robot37_policy import build_runtime, Robot37Joint96Session
from eval.robotwin.ipc import recv_message, send_message
from eval.shared_policy_server import MultiClientSessionServer, SerialRequestScheduler, SessionFactory, SessionRegistry


class AuditedSession(Robot37Joint96Session):
    def __init__(self, runtime, config, output, client_id):
        super().__init__(runtime, config)
        self.audit_output = output
        self.audit_client_id = client_id
        self.first_action_recorded = False

    def handle(self, command, payload):
        result = super().handle(command, payload)
        if command == 'get_action' and not self.first_action_recorded:
            action = np.asarray(result)
            assert action.ndim == 2 and action.shape[1] == 14 and np.isfinite(action).all()
            record = dict(client_id=self.audit_client_id, server_pid=os.getpid(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                action_shape=list(action.shape), finite=True,
                isolated_session=True, inference_seed=self.seed, replan=self.replan_steps,
                denoise=self.num_inference_steps)
            record['inference_current_visual_mode'] = 'repeat9_full_stvae'
            temp = self.audit_output / f'.FIRST_ACTION.{os.getpid()}.tmp'
            temp.write_text(json.dumps(record, indent=2) + '\n')
            temp.replace(self.audit_output / 'FIRST_ACTION.json')
            self.first_action_recorded = True
        return result


class Factory(SessionFactory):
    def __init__(self, runtime, config, output):
        self.runtime, self.config, self.output = runtime, config, output

    def create(self, client_id, metadata):
        task = metadata['task_or_split_label']
        assert task and '/' not in task and (self.output / task).is_dir()
        return AuditedSession(self.runtime, self.config, self.output / task, client_id)


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkpoint', 'socket', 'output-dir', 'run-output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--inference-config', '--training-config', dest='inference_config',
                        help='Source inference config; --training-config is a legacy alias')
    parser.add_argument('--dataset-stats', help='Default: source normalization statistics')
    parser.add_argument('--expected-step', type=int, default=None, help='Deprecated; ignored.')
    parser.add_argument('--denoise', type=int, default=10)
    parser.add_argument('--replan', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    a = parser.parse_args()
    assert a.denoise > 0 and 1 <= a.replan <= 32
    sock = Path(a.socket)
    assert not sock.exists() and len(str(sock).encode()) < 100
    runtime, config = build_runtime(a.checkpoint, a.inference_config, a.dataset_stats,
        expected_step=a.expected_step, output_dir=a.output_dir, denoise=a.denoise,
        replan=a.replan, seed=a.seed, current_visual_mode='repeat9_full_stvae')
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server = None
    try:
        listener.bind(str(sock)); os.chmod(sock, 0o600); listener.listen(2)
        scheduler = SerialRequestScheduler(max_pending=2, request_timeout_sec=600)
        server = MultiClientSessionServer(listener=listener, recv_message=recv_message,
            send_message=send_message, registry=SessionRegistry(Factory(runtime, config, Path(a.run_output)), max_sessions=2),
            scheduler=scheduler, server_id=f'phoenix-robotwin-{os.getpid()}', persistent=True,
            health_path=Path(a.output_dir) / 'SERVER_HEALTH.json', health_interval_sec=5,
            server_metadata=dict(benchmark='robotwin', variant='robot37_joint96',
                shared_runtime=True, serial_inference=True, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        signal.signal(signal.SIGTERM, lambda *_: server.request_stop())
        signal.signal(signal.SIGINT, lambda *_: server.request_stop())
        print('MODEL_SERVER_READY', str(sock), flush=True)
        server.serve()
        if scheduler.fatal_error:
            raise RuntimeError(str(scheduler.fatal_error))
    finally:
        if server:
            server.request_stop()
            print(json.dumps(server.health_snapshot()), flush=True)
        listener.close()
        sock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
