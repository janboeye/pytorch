import torch
import pickle
from torch.testing._internal.common_utils import (run_tests, TestCase)

class TestAutocastMPS(TestCase):
    def _grad_scaling_autocast_test(self, *, atol=1e-3, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
        try_pickle = False

        # 设置模型和数据
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10)
        ).to('cuda')
        data = [(
            torch.randn(10, 10, device='cuda'),
            torch.randn(10, 10, device='cuda')
        ) for _ in range(5)]
        loss_fn = torch.nn.MSELoss()
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer_ctor(model.parameters(), **optimizer_kwargs)
        scaler = torch.cuda.amp.GradScaler()

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.autocast('cuda', enabled=try_scaling_api):
                    output = model(input)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        with torch.no_grad():
                            model[1].weight.grad.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # 分别运行带和不带自动混合精度和梯度缩放的测试
        scaler = run(data, model, optimizer, scaler, loss_fn, skip_iter=2, try_scaling_api=True)
        run(data, model, optimizer, scaler, loss_fn, skip_iter=2, try_scaling_api=False)

    def test_grad_scaling_autocast(self):
        for optimizer_ctor in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            self._grad_scaling_autocast_test(optimizer_ctor=optimizer_ctor)

    # def test_grad_scaling_autocast_fused(self):
    #     for optimizer_ctor in (torch.optim.Adam, torch.optim.AdamW):
    #         self._grad_scaling_autocast_test(optimizer_ctor=optimizer_ctor, optimizer_kwargs={"fused": True})

if __name__ == "__main__":
    run_tests()
