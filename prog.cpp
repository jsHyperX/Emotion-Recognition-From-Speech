
/****************************************************************************************************************
            1.The waveInOpen function opens the given waveform-audio input device for recording.
            2.The WAVEFORMATEX structure defines the format of waveform-audio data. Only format information common
                to all waveform-audio data formats is included in this structure
            3.The WAVEHDR structure defines the header used to identify a waveform-audio buffer.
            4.The waveOutOpen function opens the given waveform-audio output device for playback.
            5.The waveOutWrite function sends a data block to the given waveform-audio output device.
*****************************************************************************************************************/
#include <Windows.h>
#include <bits/stdc++.h>
using namespace std;
short int waveIn[44100*20];
int sampleRate = 44100;         // sampling rate
HWAVEIN      hWaveIn;
WAVEHDR      WaveInHdr;
MMRESULT result;
WAVEFORMATEX pFormat;
void PlayRecord(int);
void init(int n) {
    const int NUMPTS = 44100*n;     // amount of time the audio needs to be recorded
    pFormat.wFormatTag=WAVE_FORMAT_PCM;     // simple, uncompressed format
    pFormat.nChannels=2;                    //  1=mono, 2=stereo
    pFormat.nSamplesPerSec=sampleRate;      // 44100
    pFormat.nAvgBytesPerSec=sampleRate*4;   // = nSamplesPerSec * n.BlockAlign
    pFormat.nBlockAlign=4;                  // = n.Channels * wBitsPerSample/8
    pFormat.wBitsPerSample=16;              //  16 for high quality, 8 for telephone-grade
    pFormat.cbSize=0;
    // Specify recording parameters
    result = waveInOpen(&hWaveIn, WAVE_MAPPER,&pFormat,0L, 0L, WAVE_FORMAT_DIRECT);
 // Set up and prepare header for input
    WaveInHdr.lpData = (LPSTR)waveIn;
    WaveInHdr.dwBufferLength = NUMPTS*2;
    WaveInHdr.dwBytesRecorded=0;
    WaveInHdr.dwUser = 0L;
    WaveInHdr.dwFlags = 0L;
    WaveInHdr.dwLoops = 0L;
    waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
}
void StartRecord(int n) {
 // Insert a wave input buffer
    result = waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
 // Commence sampling input
    result = waveInStart(hWaveIn);
    cout << "recording..." << endl;
    Sleep(n*1000);
 // Wait until finished recording
    waveInClose(hWaveIn);
    PlayRecord(n);
}
void PlayRecord(int n) {
    HWAVEOUT hWaveOut;
    cout << "playing..." << endl;
    waveOutOpen(&hWaveOut, WAVE_MAPPER, &pFormat, 0, 0, WAVE_FORMAT_DIRECT);
    waveOutWrite(hWaveOut, &WaveInHdr, sizeof(WaveInHdr)); // Playing the data
    Sleep(n*1000); //Sleep for as long as there was recorded
    waveInClose(hWaveIn);
    waveOutClose(hWaveOut);
}
int main() {
    int n;
    cin >> n;
    assert(n<20);
    init(n);
    StartRecord(n);
    return 0;
}
