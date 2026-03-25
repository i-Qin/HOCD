import time
import random
import pygame
import numpy as np

from utils.data import ControlInfo, SimulatorObservation

class PygameDisplay:
    def __init__(self, config):

        self._load_config(config)
        self._setup()
    
    def _load_config(self, config):
        self._cfg = config.controller.lateral
        self._lat_control_type = self._cfg.type
        self._driver_model = self._cfg.SharedControl.driver_model
    
    def _setup(self):
        # Initialize pygame and display settings
        pygame.init()
        self.width, self.height = 1920, 1080
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Carla Visualization")
        self.font = pygame.font.Font(None, 60)
        self.clock = pygame.time.Clock()
        self.fps_image = None
        self.top_image = None
        self.reaction_time_list = []

        # Button definitions
        self.button_keys = ["triangle", "rect", "circle", "times"]
        # Button states: initialized to gray (inactive)
        self.button_states = {'triangle': 'gray', 'rect': 'gray', 'circle': 'gray', 'times': 'gray'}
        # Record the time when each button is lit (activated)
        self.button_light_times = {'triangle': None, 'rect': None, 'circle': None, 'times': None}
        # Record the time when each button is turned off
        self.button_times = {'triangle': None, 'rect': None, 'circle': None, 'times': None}
        # Button positions on the screen
        self.button_positions = {
            'triangle': (self.width - 200, self.height - 300),
            'rect': (self.width - 300, self.height - 200),
            'circle': (self.width - 100, self.height - 200),
            'times': (self.width - 200, self.height - 100)
        }

        # Timer for randomly lighting buttons
        self.last_light_time = time.time()
        self.light_interval = 10

    def run_step(self, env: SimulatorObservation, ctrl: ControlInfo):
        self.screen.fill((255, 255, 255))
        self._draw_fps_image(env.image)
        reaction_time = None

        # Reaction time test only when G29 is connected
        if ctrl.lat.g29_connected:
            self._random_light_button()
            self._draw_buttons()
            reaction_time = self.handle_events(ctrl)

        pygame.display.flip()
        self.clock.tick(30)
        
        # Save reaction time into control info if available
        if reaction_time is not None:
            ctrl.lat.reaction_time = reaction_time
            print("reaction_time", reaction_time)

    def _draw_fps_image(self, fps_image):
        if fps_image is not None:
            fps_img_surface = pygame.surfarray.make_surface(np.transpose(fps_image, (1, 0, 2)))
            fps_img_surface = pygame.transform.scale(fps_img_surface, (self.width, self.height))
            self.screen.blit(fps_img_surface, (0, 0))
        else:
            print("FPS image is None")

    def _random_light_button(self):
        if time.time() - self.last_light_time > self.light_interval:
            button_to_light = random.choice(['triangle', 'rect', 'circle', 'times'])
            # Only activate if currently inactive
            if self.button_states[button_to_light] == 'gray': 
                self.button_states[button_to_light] = 'green'
                self.button_light_times[button_to_light] = time.time() 
            # Update last activation time
            self.last_light_time = time.time() 

    def _draw_buttons(self):
        for button_id, pos in self.button_positions.items():
            if self.button_states[button_id] == 'gray':
                color = (169, 169, 169)  
            elif self.button_states[button_id] == 'green':
                color = (144, 238, 144) 
            
            pygame.draw.circle(self.screen, color, pos, 50)
            
            label = ''
            if button_id == 'triangle':
                label = 'image/pygame_triangle.png'
            elif button_id == 'rect':
                label = 'image/pygame_rect.png'
            elif button_id == 'circle':
                label = 'image/pygame_circle.png'
            elif button_id == 'times':
                label = 'image/pygame_times.png'
            
            img = pygame.image.load(label)
            self.screen.blit(img, (pos[0]-30, pos[1]-30))  # 放在右下角

    def handle_events(self, ctrl: ControlInfo):
        """Handle user input and return reaction time if triggered"""
        for key in self.button_keys:
            if getattr(ctrl.lat, key):
                reaction_time=self._handle_button_click(key)
                if reaction_time is not None:
                    return reaction_time
        return None

    def _handle_button_click(self, button_id):
        """Handle button press and compute reaction time"""
        if self.button_states[button_id] == 'green':
             # Compute time from activation to response
            if self.button_light_times[button_id]:
                elapsed_time = time.time() - self.button_light_times[button_id]
                print(f"Button {button_id} turned off after {elapsed_time:.2f} seconds.")
            # Reset button state
            self.button_states[button_id] = 'gray'  
            self.button_times[button_id] = time.time() 
            self.button_light_times[button_id] = None 
            return elapsed_time
        else:
            return None


