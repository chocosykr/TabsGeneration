import torch
from karplus_strong import DifferentiableKarplusStrong
import scipy.io.wavfile as wavfile
from optimization import multi_scale_stft_loss

sample_rate, target = wavfile.read('test_E4.wav')
target = torch.from_numpy(target).float() / 32767.0
target = target.unsqueeze(0)

model = DifferentiableKarplusStrong()
model.pitch.copy_(torch.tensor(329.87 - 20))
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

for i in range(10):
    optimizer.zero_grad()
    y = model()
    loss = multi_scale_stft_loss(y, target)
    print(f"Iter {i}, Loss: {loss.item()}, Pitch: {model.pitch.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if torch.isnan(model.pitch):
        print("PITCH IS NAN!")
        break
