from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
from math import sin, acos, degrees
import numpy as np
import matplotlib as plt


class BestController(KesslerController):


    def __init__(self):
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-30])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90,-30,0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-30,0,30])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,30,90])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 

        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)


        ## Fuzzy system 2

        threshold_distance = 100;

        distance = ctrl.Antecedent(np.arange(-200, 200, 1), 'distance')
        relative_speed = ctrl.Antecedent(np.arange(-300, 300, 1), 'relative_speed')
        thrust = ctrl.Consequent(np.arange(-450, 450, 1), 'thrust')

        distance['very_close'] = fuzz.trimf(distance.universe, [-100, 0, threshold_distance/3])
        distance['close'] = fuzz.trimf(distance.universe, [0, threshold_distance/3, 2*threshold_distance/3])
        distance['medium'] = fuzz.trimf(distance.universe, [threshold_distance/3, 2*threshold_distance/3, threshold_distance])
        distance['far'] = fuzz.trimf(distance.universe, [2*threshold_distance/3, threshold_distance, threshold_distance])

        relative_speed['slow'] = fuzz.trimf(relative_speed.universe, [-200, -100, 0])
        relative_speed['zero'] = fuzz.trimf(relative_speed.universe, [-100, 0, 100])
        relative_speed['fast'] = fuzz.trimf(relative_speed.universe, [0, 100, 200])

        thrust['negative'] = fuzz.trimf(thrust.universe, [-300, -250, 0])
        thrust['zero'] = fuzz.trimf(thrust.universe, [-250, 0, 250])
        thrust['positive'] = fuzz.trimf(thrust.universe, [0, 250, 300])

        # Define rules
        rule1_ = ctrl.Rule(distance['very_close'] & relative_speed['slow'], thrust['negative'])
        rule2_ = ctrl.Rule(distance['very_close'] & relative_speed['zero'], thrust['negative'])
        rule3_ = ctrl.Rule(distance['very_close'] & relative_speed['fast'], thrust['negative'])

        rule4_ = ctrl.Rule(distance['close'] & relative_speed['slow'], thrust['negative'])
        rule5_ = ctrl.Rule(distance['close'] & relative_speed['zero'], thrust['negative'])
        rule6_ = ctrl.Rule(distance['close'] & relative_speed['fast'], thrust['negative'])
        
        rule7_ = ctrl.Rule(distance['medium'] & relative_speed['slow'], thrust['negative'])
        rule8_ = ctrl.Rule(distance['medium'] & relative_speed['zero'], thrust['negative'])
        rule9_ = ctrl.Rule(distance['medium'] & relative_speed['fast'], thrust['negative'])

        rule10_ = ctrl.Rule(distance['far'] & relative_speed['slow'], thrust['positive'])
        rule11_ = ctrl.Rule(distance['far'] & relative_speed['zero'], thrust['positive'])
        rule12_ = ctrl.Rule(distance['far'] & relative_speed['fast'], thrust['positive'])

        # Create control system
        self.targeting_control2 = ctrl.ControlSystem()
        self.targeting_control2.addrule(rule1_)
        self.targeting_control2.addrule(rule2_)
        self.targeting_control2.addrule(rule3_)
        self.targeting_control2.addrule(rule4_)
        self.targeting_control2.addrule(rule5_)
        self.targeting_control2.addrule(rule6_)
        self.targeting_control2.addrule(rule7_)
        self.targeting_control2.addrule(rule8_)
        self.targeting_control2.addrule(rule9_)
        self.targeting_control2.addrule(rule10_)
        self.targeting_control2.addrule(rule11_)
        self.targeting_control2.addrule(rule12_)



    def calculate_adjusted_angle(self, ship_state, future_asteroid_position, asteroid_theta, ship_ast_dist, asteroid_velocity, bullet_speed):
        #ship_velocity = np.array(ship_state['velocity'])

        future_dist = sin(asteroid_theta)*ship_ast_dist

        theta_angle_adjusted = degrees(acos(future_dist/ship_ast_dist))

        future_dist = np.sin(asteroid_theta) * ship_ast_dist

        # Calculate the angle between ship and asteroid using arccos
        ship_to_asteroid_vector = np.array(future_asteroid_position) - np.array(ship_state['position'])
        cos_theta = np.dot(ship_to_asteroid_vector, np.array([1, 0])) / (np.linalg.norm(ship_to_asteroid_vector) * np.linalg.norm(np.array([1, 0])))
        theta_angle = np.degrees(np.arccos(cos_theta))

        # Adjust the sign based on the orientation
        if ship_to_asteroid_vector[1] < 0:
            theta_angle = -theta_angle
        
        # Calculate the adjusted angle
        #theta_angle_adjusted = theta_angle

        
        return theta_angle_adjusted

    def calc_fire_target(self, a, ship_state):
        
        target_loc = np.array(a['position'])
        ship_loc = np.array(ship_state['position'])
        target_vel = np.array(a['velocity'])

        # calc dist to target
        dist_vec = np.array(ship_loc - target_loc)
        dist = np.linalg.norm(dist_vec)
        # Cacl time for bullet to hit target
        # Bullet speed is 800
        bullet_speed = 800
        time_to_hit = dist/bullet_speed
        # calc loc target will be at when bullet arrives at original target pos
        new_target_loc = target_loc + time_to_hit * target_vel
        # return loc.

        print('target', target_loc, 'new_target', new_target_loc)
        return new_target_loc

    def has_asteroids_in_path(self, ship_state, turn_rate):
        threshold = 20
        
        ship_angle = ship_state['heading']-180

        if turn_rate>0:
            turn_rate/=1
        print('ship_head', ship_angle)
        print('turn_rate', turn_rate)
        if(abs(turn_rate)<10):
            return True
        else:
            False


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:

        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)

            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object.
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!


        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].

        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        # print("bullet time" , bullet_t)
        # print("theta delta" , shooting_theta)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        shooting.compute()
        future_asteroid_position = np.array([intrcpt_x, intrcpt_y])
        theta_angle_adjusted = self.calculate_adjusted_angle(ship_state, future_asteroid_position, asteroid_ship_theta, closest_asteroid['dist'], asteroid_vel, 800)


        '''
        print('Adjusted angle:', theta_angle_adjusted)
        print('ship_head', ship_state['heading']-180)

        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        print("tr:",turn_rate)

        
        if turn_rate <0:
            turn_rate = -(theta_angle_adjusted*1.7)

        else:
            turn_rate = theta_angle_adjusted*1.7

        #turn_rate = theta_angle_adjusted

        '''
        target_loc = self.calc_fire_target(closest_asteroid["aster"], ship_state)
        angle = math.atan2(target_loc[1] - ship_state['position'][1], target_loc[0] - ship_state['position'][0]) * 180/math.pi
        if angle < 0:
            angle += 360

        diff = ship_state['heading'] - angle
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 180

        print(diff)
        turn_rate = -diff * 30
        cond = self.has_asteroids_in_path(ship_state, -diff)
        print(cond)
        fire_radius = 800
        if (closest_asteroid["dist"] <= fire_radius) and (cond == True):
            fire = True
        else:
            fire = False

        ship_radius = 20
        threshold_distance = 80

        max_threat_asteroid_px = closest_asteroid["aster"]["position"][0]
        max_threat_asteroid_py = closest_asteroid["aster"]["position"][1]
        max_threat_asteroid_vx = closest_asteroid["aster"]["velocity"][0]
        max_threat_asteroid_vy = closest_asteroid["aster"]["velocity"][1]
        map_width, map_height = 1000, 800
        max_possible_distance = math.sqrt(map_width**2 + map_height**2)  
        asteroid_radius = closest_asteroid["aster"]["size"]*8

        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = 0.0

        if closest_asteroid["dist"] < threshold_distance:

            vector_to_asteroid = np.array([max_threat_asteroid_px - ship_pos_x, max_threat_asteroid_py - ship_pos_y])


            normalized_distance = (closest_asteroid["dist"]  - ship_radius - asteroid_radius)

            relative_speed = np.dot(np.array([max_threat_asteroid_vx, max_threat_asteroid_vy]), vector_to_asteroid) / closest_asteroid["dist"]
        
            thrust_fuzzy = ctrl.ControlSystemSimulation(self.targeting_control2,flush_after_run=1)
            thrust_fuzzy.input['distance'] = normalized_distance
            thrust_fuzzy.input['relative_speed'] = relative_speed  # You need to replace this with the actual value
            thrust_fuzzy.compute()




            thrust = thrust_fuzzy.output['thrust']

        self.eval_frames +=1

        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        drop_mine =  False

        if (ship_state['is_respawning'] == True):
            fire = 0
            thrust =  thrust


        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "watch"
