# test sound
from pygame import mixer
mixer.init() 
sound=mixer.Sound("bell.wav")
sound.play()