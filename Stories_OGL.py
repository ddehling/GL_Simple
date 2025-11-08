import numpy as np
import time
from pathlib import Path
from corefunctions.Events import EventScheduler

from corefunctions.soundinput import MicrophoneAnalyzer
from corefunctions.weather_params import WeatherState, DEFAULT_WEATHER_PARAMS, WEATHER_PRESETS
from corefunctions.shader_effects.shader_fog import ShaderFog
from corefunctions.shader_effects.celestial_bodies import (
     shader_celestial_bodies, 
     CELESTIAL_BODIES
 )
from corefunctions import shader_effects as fx

class EnvironmentalSystem:
    def __init__(self, scheduler):
        frame_dimensions = [
            (256, 300),   # Frame 0 (primary/main display)
              # Frame 1 (secondary display)
        ]
        self.scheduler = EventScheduler(
        use_shader_renderer=True,
        headless=False,frames=frame_dimensions
    )
        self.current_weather = WeatherState.CLEAR
        self.target_weather = WeatherState.CLEAR
        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.analyzer = MicrophoneAnalyzer(device_name="TONOR")
        self.analyzer.start()
        #self.specdat = np.zeros([513, 1000])
        
        # Initialize celestial bodies
        self.celestial_bodies = CELESTIAL_BODIES.copy()
        # sort celestial bodies by distance, farthest first
        self.celestial_bodies.sort(key=lambda x: x.distance, reverse=True)

        # Keep track of active weather effects
        self.default_weather_params = DEFAULT_WEATHER_PARAMS.copy()
        self.weather_params = self.default_weather_params.copy()

        # Weather state parameters
        self.weather_presets = WEATHER_PRESETS
        self.scheduler.state["tree"] = False
        self.scheduler.state["skyfull"] = False
        self.scheduler.state["simulate"] = True  # Display the leds in an opencv window for visualization
        self.active_effects = {"world": None, "ambient_sound": None}
        self._prewarm_audio_cache()
        corners_frame0 = [
        (-18, 18),  # Top-left
        (18, 18),   # Top-right
        (18, 0),    # Bottom-right
        (-18, 0)    # Bottom-left
    ]
    
    # Define corners for frame 1 (upward-facing view)
        corners_frame1 = [
        (-0, 40),  # Top-left
        (180, 40),   # Top-right
        (180, 80),   # Bottom-right
        (-0, 80)   # Bottom-left
    ]
        sim_forever=999999999
        self.scheduler.schedule_event(0, sim_forever, fx.shader_drifting_clouds, frame_id=0)
        self.scheduler.schedule_event(0, sim_forever, fx.shader_firefly,frame_id=0)
        self.scheduler.schedule_event(0, sim_forever, fx.shader_stars,frame_id=0)
        # self.scheduler.schedule_event(0, sim_forever, fx.shader_celestial_bodies, 
        #                     corners=corners_frame0, frame_id=0)
        self.scheduler.schedule_event(0, sim_forever, fx.shader_rain, frame_id=0)
        viewport0 = self.scheduler.shader_renderer.get_viewport(0)
        if viewport0:
            fog0 = viewport0.add_effect(ShaderFog, 
                                        strength=0.0, 
                                        color=(0.5, 0.6, 0.8),  # Blue-ish fog
                                        fog_near=20.0,
                                        fog_far=80.0)

        # Schedule world rendering events for each frame, keeping the original function names
        #self.active_effects["world"] = self.scheduler.schedule_event(0, 999999999, multilayer_world, frame_id=0) # noqa: F405
        #self.active_effects["secondary_world"] = self.scheduler.schedule_event(0, 999999999, secondary_multilayer_world, frame_id=1) # noqa: F405
        self.whompcount = 0

    def _prewarm_audio_cache(self):
        """Pre-warm the audio cache with all weather sound effects"""
        print("Pre-warming audio cache...")
        for weather_state, params in self.weather_presets.items():
            if "ambient_sound" in params:
                sound_path = Path("media") / Path("sounds") / params["ambient_sound"]
                duration = self.get_weather_params(weather_state).get("ARI", 40)
                skip_time = self.get_weather_params(weather_state).get("skiptime", 0)
                volume = self.get_weather_params(weather_state).get("Sound_volume", 0)
                try:
                    # This will automatically cache the audio in AudioCache
                    self.scheduler.state["soundengine"].load_audio(
                        sound_path, duration, skip_time, volume
                    )
                    print(f"Cached: {sound_path.name}")
                except Exception as e:
                    print(f"Failed to cache {sound_path.name}: {str(e)}")

    def get_weather_params(self, weather_state: WeatherState):
        """Get the complete set of parameters for a weather state by combining with defaults"""
        params = self.default_weather_params.copy()
        params.update(self.weather_presets[weather_state])
        return params

    def transition_to_weather(self, new_weather: WeatherState, transition_duration: float = 10.0):
        """Start a transition to a new weather state"""
        self.target_weather = new_weather
        print(self.target_weather)
        self.transition_time = transition_duration
        self.transition_start = time.time()

        # Start new effects if needed, one offs that occur when a weather state happens
        target_params = self.get_weather_params(new_weather)
        
        # Schedule weather-specific events for appropriate frames, using original function names
        # if new_weather == WeatherState.VOLCANO:
        #     self.scheduler.schedule_event(0, 100, volcanic_mountain, frame_id=0) # noqa: F405

        # if new_weather == WeatherState.SANDSTORM:
        #     self.scheduler.schedule_event(0, 100, sandstorm, frame_id=0) # noqa: F405
        #     self.scheduler.schedule_event(0, 100, secondary_sandstorm, frame_id=1) # noqa: F405

        # if new_weather == WeatherState.ASTEROID:
        #     self.scheduler.schedule_event(0, 20, meteor_shower, frame_id=0) # noqa: F405
        #     self.scheduler.schedule_event(0, 30, secondary_meteor_shower, frame_id=1) # noqa: F405
        #     self.scheduler.schedule_event(0, 30, secondary_alarm, frame_id=1) # noqa: F405
        #     sound_path = (Path("media") / Path("sounds") / "45. Buzzer - 'Space Alarm' Warning.flac")
        #     self.scheduler.state["soundengine"].schedule_event(sound_path, time.time(), 20)

        # if new_weather == WeatherState.HEAVY_FOG:
        #     self.scheduler.schedule_event(0, 80, fx.shader_chromatic_fog_beings, frame_id=0) # noqa: F405

        # # if new_weather == WeatherState.MUSHROOM:
        # #     if not self.scheduler.state.get("has_mushrooms", False):
        # #         self.scheduler.schedule_event(0, 100, growing_mushrooms, frame_id=0) # noqa: F405
        # #         if self.scheduler.state.get("has_clouds", False):
        # #             self.scheduler.state["has_clouds"] = True
        # #             self.scheduler.schedule_event(70, 40, drifting_clouds, frame_id=0) # noqa: F405

        # if new_weather == WeatherState.FALLING_LEAVES:
        #     if not self.scheduler.state.get("has_leaves", False):
        #         self.scheduler.schedule_event(0, 60, fx.shader_falling_leaves, frame_id=0) # noqa: F405


        # if new_weather == WeatherState.SUMMER_BLOOM:
        #     self.scheduler.schedule_event(0, 90, bioluminescent_wildflowers, frame_id=0) # noqa: F405

        # Handle ambient sound transition
        if self.active_effects["ambient_sound"]:
            # Fade out the currently playing sound
            self.scheduler.state["soundengine"].fade_out_audio(self.active_effects["ambient_sound"], 5)

        # Schedule new ambient sound
        sound_path = Path("media") / Path("sounds") / target_params["ambient_sound"]
        self.active_effects["ambient_sound"] = target_params["ambient_sound"]
        self.scheduler.state["soundengine"].schedule_event(
            sound_path,
            time.time(),
            target_params["ARI"],
            repeat_interval=target_params["ARI"],
            inname=self.active_effects["ambient_sound"],
            fade_in_duration=5.0,
            skip_time=target_params["skiptime"],
        )

    def calculate_seasonal_weight_multiplier(self, season_preference, current_season):
        """
        Calculate a weight multiplier based on how close the current season is to the preferred season.
        Returns a value between 0.5 (furthest from preferred) and 3.0 (at preferred season).
        """
        # Calculate distance between current season and preferred season
        # Since seasons are cyclical (0-1), we need to find the shortest distance
        distance = abs(current_season - season_preference)
        if distance > 0.5:
            distance = 1.0 - distance  # Take the shorter path around the cycle
        
        # Normalize distance to range [0, 1] where 0 means perfect match and 1 means opposite season
        normalized_distance = distance * 2  # Now 0 = perfect match, 1 = opposite season
        
        # Calculate multiplier that varies from 3.0 (perfect match) to 0.5 (opposite season)
        multiplier = 1.0 - (normalized_distance * .95)
        
        return multiplier

    # def get_whomp(self):
    #     thresh = 1.0
    #     maxsound = 6
    #     # loud = self.analyzer.get_sound()
    #     loud = self.analyzer.get_all_sound()
    #     swloud = (loud > thresh) * 1
    #     self.whomp = swloud * (np.clip(loud, 0, maxsound) - thresh) / (maxsound - thresh)

    def transition_update(self):
        # self.progress = 1.0
        if self.current_weather != self.target_weather:
            self.progress = min(
                1.0, (self.current_time - self.transition_start) / self.transition_time
            )

            start_params = self.get_weather_params(self.current_weather)
            target_params = self.get_weather_params(self.target_weather)

            # Interpolate parameters
            for param in target_params:
                if isinstance(target_params[param], (int, float, np.ndarray)):
                    self.weather_params[param] = (
                        target_params[param] - start_params[param]
                    ) * self.progress + start_params[param]
                else:
                    # For non-numeric parameters, just use the target value
                    self.weather_params[param] = target_params[param]

            if self.progress >= 1.0:
                self.current_weather = self.target_weather
                self.weather_params = target_params.copy()

    def send_variables(self):
        self.season = (time.time() / 1800) % 1
        fog = np.maximum(0,self.weather_params["fog"] * (0.75 - 0.25 * np.cos(np.pi * 2 * (self.season - 0.625))))
        self.cloudyness = (1 - self.weather_params["starryness"]) + (1 - self.weather_params["celestial_visibility"]) + fog + self.weather_params["rain_rate"] + self.weather_params["wind_speed"] / 3
        self.scheduler.state["cloudyness"] = self.cloudyness
        # Set variables in scheduler state
        self.scheduler.state["fog_strength"] = fog
        self.scheduler.state["fog_color"] = self.weather_params["fog_color"]
        #self.scheduler.state["whomp"] = self.whomp
        self.scheduler.state["wind"] =self.weather_params["wind_speed"] * ( np.cos(np.pi * 2 * (self.season - 0.125)))  # noqa: F405
        self.scheduler.state["season"] = self.season
        self.scheduler.state["rain"] = self.weather_params["rain_rate"]
        self.scheduler.state["starryness"] = self.weather_params["starryness"]
        self.scheduler.state["sound"] = self.analyzer.get_extended_analysis()
        # # Set fog for each frame
        # self.scheduler.set_fog(0, fog, tuple(self.weather_params["fog_color"]), dir_scale=(1.0, 1.0))
        # self.scheduler.set_fog(1, fog + self.cloudyness / 2, tuple(self.weather_params["fog_color"]), dir_scale=(1.0, 0.0))
        
        self.scheduler.state["celestial_bodies"] = self.celestial_bodies
        self.scheduler.state["celestial_visibility"] = self.weather_params["celestial_visibility"]
        self.scheduler.state["firefly_density"] = self.weather_params["firefly_density"]
        self.scheduler.state["meteor_rate"] = self.weather_params["meteor_rate"]
        #print(self.weather_params["meteor_rate"])
        self.scheduler.state["volcano_level"] = (np.sin(self.current_time / 100) * 0.5 + 0.5) * self.weather_params["volcano_level"]
        self.scheduler.state["sand_density"] = self.weather_params.get("sand_density", 0)
        self.scheduler.state["tree_growth"] = (self.weather_params.get("tree_prob", 0) + 0.25)

    def random_events(self):
        randcheck = np.random.random()
        
        # if (randcheck < self.cloudyness / 1000):
        #     if not self.scheduler.state.get("has_clouds", False):
        #         self.scheduler.schedule_event(0, 100, fx.shader_drifting_clouds, frame_id=0) # noqa: F405

        # if (randcheck < self.weather_params["Weird"] / 10000):
        #     # Choose between different options
        #     sw = np.random.randint(0, 9)
        #     if sw < 8:
        #         self.scheduler.schedule_event(0, 40, psychedelic_spiral, frame_id=0) # noqa: F405
        #         self.scheduler.schedule_event(0, 40, secondary_psychedelic_spiral, frame_id=1) # noqa: F405
        #     elif sw == 9:
        #         self.scheduler.schedule_event(0, 60, fluid_pond, frame_id=0) # noqa: F405
        #     elif sw == 8:
        #         self.scheduler.schedule_event(0, 60, colorful_conway, frame_id=0) # noqa: F405

        # if (randcheck < self.weather_params["tree_prob"] / 400) & (not self.scheduler.state["tree"]):
        #     # self.scheduler.schedule_event(0, 100, secondary_tree, frame_id=1) # noqa: F405
        #     self.scheduler.schedule_event(0, 100, fx.shader_forest, frame_id=0) # noqa: F405
        
        # Wolf howl
        # if (randcheck < (self.weather_params["Wolfy"] + self.weather_params["spookyness"] / 10) / 2000):
        #     self.scheduler.schedule_event(0, 10, Awooo_Wolf_Howl, frame_id=0) # noqa: F405

        # # Giant auroras in the sky
        # if randcheck < self.weather_params["Aurora_probability"] / 1000:
        #     self.scheduler.schedule_event(0, 50, fx.shader_aurora, frame_id=0) # noqa: F405
        #     #self.scheduler.schedule_event(0, 50, secondary_Aurora, frame_id=1) # noqa: F405

        # if randcheck < (1 + np.clip(self.whomp, 0, 2)) * self.weather_params["lightning_probability"] / 250:
        #     # Choose between primary and secondary lightning
        #     if np.random.random() < 0.5:
        #         self.scheduler.schedule_event(0, 1, fx.shader_lightning, frame_id=0) # noqa: F405
        #     # else:
        #     #     self.scheduler.schedule_event(0, 10, secondary_lightning, frame_id=1) # noqa: F405
                
        randcheck = np.random.random()

        # # Sand storms
        # if randcheck < self.weather_params["sand_density"] / 2000:
        #     self.scheduler.schedule_event(0, 45, sandstorm, frame_id=0) # noqa: F405
        #     self.scheduler.schedule_event(0, 45, secondary_sandstorm, frame_id=1) # noqa: F405

        # # Spooky giant eye
        # if randcheck < self.weather_params["spookyness"] / 800:
        #     self.scheduler.schedule_event(0, 30, fx.shader_eye, frame_id=0) # noqa: F405

        # # Random meteor events
        # if randcheck < self.weather_params["meteor_rate"] / 800:
        #     self.scheduler.schedule_event(0, 25, fx.shader_meteor, frame_id=0,direction='bottom') # noqa: F405
        #     #self.scheduler.schedule_event(0, 25, secondary_meteor_shower, frame_id=1) # noqa: F405

        # Dancing cactus events
        randcheck = np.random.random()
        # if randcheck < (self.weather_params["Weird"] / 5000 + self.weather_params["spookyness"] / 4000 + 1 / 20000):
        #     if not self.scheduler.state.get("has_cactus", False):
        #         # Define all possible cactus types and their corresponding functions
        #         #cactus_types = [dancing_cactuses, dancing_joshua, dancing_prickly_pear, dancing_barrel_cactus]  # noqa: F405
        #         cactus_types = [fx.shader_cactus, fx.shader_cactus, fx.shader_cactus, fx.shader_cactus]  # noqa: F405
        #         weights = [1 * (1 + self.weather_params["spookyness"]), 1, 1, 1]  # Equal probability for each
        #         weights = weights / np.sum(weights)
        #         selected_cactus = np.random.choice(cactus_types, p=weights)
        #         self.scheduler.schedule_event(0, 100, selected_cactus, frame_id=0)

        # if randcheck < (self.weather_params["mountain"] / 1500):
        #     if not self.scheduler.state.get("has_mountain", False):
        #         self.scheduler.schedule_event(0, 180, mountain_scene, frame_id=0) # noqa: F405
                
    def random_state_change(self):
        randcheck = np.random.random()
        if (randcheck < (1 / 800) * self.weather_params["Switch_rate"]) and (self.progress >= 0.99):  # 0.1% chance each frame
            self.progress = 0
            current_preset = self.weather_presets[self.current_weather]
            possible_states = [WeatherState(state) for state in current_preset["possible_transitions"]]
            base_weights = current_preset["transition_weights"]
            #print(f"Season = {self.season:.3f}")
            
            # Apply seasonal modifiers to weights
            adjusted_weights = []
            for i, state in enumerate(possible_states):
                # Get the season preference for this weather state
                target_season_pref = self.weather_presets[state].get("season_preference", 0.375)
                
                # Calculate seasonal multiplier (between 0.5 and 3.0)
                season_multiplier = self.calculate_seasonal_weight_multiplier(target_season_pref, self.season)
                
                # Apply the seasonal modifier to the base weight
                adjusted_weight = base_weights[i] * season_multiplier
                adjusted_weights.append(adjusted_weight)
                # Print season name and adjusted weight
                #print(f"Weather: {state.name}, Base Weight:{base_weights[i]:.2f}, Adjusted Weight: {adjusted_weight:.2f}")
            
            # Normalize weights
            adjusted_weights = np.array(adjusted_weights)
            if np.sum(adjusted_weights) > 0:  # Avoid division by zero
                adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            else:
                # Fallback to equal probabilities if all weights are zero
                adjusted_weights = np.ones(len(adjusted_weights)) / len(adjusted_weights)
            
            # Choose new weather state with seasonally adjusted weights
            new_weather = np.random.choice(possible_states, p=adjusted_weights)
            
            # Find the 'transition duration of the new weather'
            new_weather_params = self.get_weather_params(new_weather)
            t_duration = new_weather_params["transition_duration"]
            self.transition_to_weather(new_weather, transition_duration=t_duration)

    def update(self):
        """Update the environmental system - should be called each frame"""
        #self.get_whomp()
        self.current_time = time.time()
       
        # OSC handling
        #messages = self.scheduler.get_osc_messages()
        #if messages != []:
        #    print(messages)  # Eventually want to pass these to the scheduler
            
        # Handle transitions
        self.transition_update()

        # Update celestial bodies
        for body in self.celestial_bodies:
            body.update(self.current_time)
            
        # Apply current parameters to scheduler state
        self.send_variables()
        
        # Random events
        self.random_events()
        self.random_state_change()
        
        # Update the scheduler
        self.scheduler.update()


# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)

    # Start with summer bloom weather
    env_system.transition_to_weather(WeatherState.FOGGY)
    env_system.scheduler.schedule_event(0, 500, fx.shader_test_circles,frame_id=0)  # noqa: F405
    env_system.scheduler.schedule_event(0, 500, fx.shader_falling_leaves,frame_id=0)
    last_time = time.time()
    FRAME_TIME = 1 / 60
    first_time = time.time()
    frame_count = 0
    try:
        while True:
            # Update environmental system
            env_system.update()

            current_time = time.time()
            elapsed = current_time - last_time
            sleep_time = max(0, FRAME_TIME - elapsed)
            time.sleep(sleep_time)
            
            frame_count += 1
            if frame_count % 50 == 0:  # Print FPS every second
                actual_fps = 1.0 / (elapsed + sleep_time)
                #um_effects = sum(len(vp.effects) for vp in scheduler.shader_renderer.viewports)
                print(f"FPS: {actual_fps:.1f}, Active events: {len(scheduler.active_events)}")
            
            last_time = current_time
            # Print stats if needed
            # print(["%.2f" % (1/(time.time()-lasttime)), "%.2f" % len(scheduler.active_events), len(scheduler.event_queue),"%.3f" %((lasttime-first_time)/3600)])
            #last_time = time.time()

    except KeyboardInterrupt:
        print("Done!")