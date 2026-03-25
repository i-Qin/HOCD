import pygame
from configparser import ConfigParser
from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q

from utils.data import LatDebugInfo


class G29:
    def __init__(self, config=None):
        pygame.init()

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        # Load configuration from config file
        self._parser = ConfigParser()
        self._parser.read('utils/wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

        self.connected = True

        # Turn signal states
        self.left_triggered = False   # whether left turn signal is active
        self.right_triggered = False  # whether right turn signal is active

        self.R_button_index = 4    # right paddle
        self.L_button_index = 5    # left paddle
        self.back_button_index = 7 # reset (mapped to L2)

        # Button mapping (PlayStation layout)
        self.times_button_index = 0
        self.rect_button_index = 1
        self.circle_button_index = 2
        self.triangle_button_index = 3

        # Button states
        self.times_state = False
        self.rect_state = False
        self.circle_state = False
        self.triangle_state = False

    def control(self, path_list=None, vehicle_state=None, e_rr=None, min_index=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                
        # Parse steering wheel input
        steer = self._parse_vehicle_wheel()

        # Infer human driving intention (e.g., lane change direction)
        intention = self._get_human_intent()
        
        return steer, LatDebugInfo(
            intention=intention,
            g29_connected=self.connected,
            triangle=self.triangle_state,
            rect=self.rect_state,
            circle=self.circle_state,
            times=self.times_state
        )

    def _get_human_intent(self) -> str:
        """
        Human intention recognition
        Returns:
            str: direction intention ("LEFT", "RIGHT", or "")
        """

        # Logic for activating/deactivating turn signals
        if self.left_state:
            if not self.left_triggered:
                # Activate left turn signal and deactivate right
                self.left_triggered = True
                self.right_triggered = False
                print("Left turn signal activated!")
        elif self.right_state:
            if not self.right_triggered:
                # Activate right turn signal and deactivate left
                self.right_triggered = True
                self.left_triggered = False
                print("Right turn signal activated!")
        elif self.back_state:
            # Reset all signals when L2 button is pressed
            if self.left_triggered or self.right_triggered:
                print("Turn signals cleared!")
            self.left_triggered = False
            self.right_triggered = False

        # Output current intention
        if self.left_triggered:
            return "LEFT"
        elif self.right_triggered:
            return "RIGHT"
        else:
            return ""

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        # Read paddle and button states
        self.left_state = bool(self._joystick.get_button(self.L_button_index))
        self.right_state = bool(self._joystick.get_button(self.R_button_index))
        self.back_state = bool(self._joystick.get_button(self.back_button_index))

        self.times_state = bool(self._joystick.get_button(self.times_button_index))
        self.rect_state = bool(self._joystick.get_button(self.rect_button_index))
        self.circle_state = bool(self._joystick.get_button(self.circle_button_index))
        self.triangle_state = bool(self._joystick.get_button(self.triangle_button_index))

        # Note: Steering wheel range should be set to 360° in G29 driver settings
        steerCmd = jsInputs[self._steer_idx] * 3.14

        return steerCmd

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


if __name__ == '__main__':
    controller = G29()
    while True:
        steer, info = controller.control()
        print(f"steer: {steer}, intention: {info.intention}")

