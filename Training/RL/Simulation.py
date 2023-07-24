import random
from gym import spaces

import gymnasium
import numpy as np
import openai
import torch

from Knowledge.ConceptExtraction import concept_extractor
from Knowledge.ConceptNet import ConceptNet
from Chatbot import TargetGuidedChatbot, Userbot, Chatbot

from sentence_transformers import SentenceTransformer, util

openai.api_key = "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"


class Simulator(gymnasium.Env):
    def __init__(self, target, verbose=False):
        super(Simulator, self).__init__()

        self.state = None
        self.state_dim = 769
        self.state_low = np.array([-np.inf] * self.state_dim)  # Replace n_dimensions with the number of dimensions
        self.state_high = np.array([np.inf] * self.state_dim)

        self.observation_space = spaces.box.Box(low=self.state_low,
                                                high=self.state_high,
                                                shape=(self.state_dim,))
        self.action_space = spaces.discrete.Discrete(677070)

        # ConceptNet
        self.conceptnet = ConceptNet('../../Knowledge/Data/conceptnet_en.txt')
        self.concepts = list(self.conceptnet.conceptnet.nodes)

        # Sentence Encoder
        self.sentence_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.target = target
        self.task_completion = False

        # Chatbot
        self.user = None
        self.chatbot = None

        self.previous_user_response = None
        self.verbose = verbose
        self.maximum_steps = 10
        self.current_steps = 0
        self.episode = 0
        self.step_reward = 0

        self.opening_messages = [
            "Hello! How can I assist you today?",
            "Welcome! How may I help you?",
            "Hi there! How can I make your day better?",
            "Greetings! How can I be of service to you?",
            "Good day! What brings you here today?",
            "Hello! How can I support you with your inquiries?",
            "Welcome! How may I provide assistance?",
            "Hi there! How can I lend a hand?",
            "Greetings! How may I guide you today?",
            "Good to see you! How can I assist you with your needs?"
        ]

        self.initialize()

    def initialize(self):
        # opening setting of the environment
        opening = random.choice(self.opening_messages)

        self.user = Userbot.Userbot()
        self.chatbot = Chatbot.Chatbot(opening)

        self.previous_user_response = self.user.chat(opening)

        context = self.context_encoder(self.chatbot.chat_history)
        response = self.sentence_encoder.encode(self.previous_user_response, convert_to_tensor=True)

        self.state = torch.cat((context, response)).detach().cpu().numpy()

    def step(self, action):
        keyword = self.concepts[action]

        tgchatbot_response = self.chatbot.chat(self.previous_user_response, keyword)
        user_response = self.user.chat(tgchatbot_response)

        if self.verbose:
            print("[Chatbot][{}] {}".format(keyword, tgchatbot_response))
            print("[Userbot] {}".format(user_response))

        done = False
        if self.is_complete(user_response):
            done = True
        self.step_reward = self.reward(user_response)

        self.previous_user_response = user_response
        context = self.context_encoder(self.chatbot.chat_history)
        response = self.sentence_encoder.encode(user_response, convert_to_tensor=True)

        self.state = torch.cat((context, response)).detach().cpu().numpy()

        self.current_steps = self.current_steps + 1

        info = self._get_info()
        return self.state, self.step_reward, done, self.current_steps > self.maximum_steps, info

    def reward(self, response):
        user_concepts = self.get_user_concepts(response)
        min_path_length = 0
        for user_concept in user_concepts:
            _, path_weight_sum = self.conceptnet.shortest_path(user_concept, self.target)
            min_path_length = min(min_path_length, path_weight_sum)
        return 1 / (min_path_length + 1e-6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.verbose:
            print('[Conversation Terminated] Task Completion :{}'.format(self.task_completion))
            print()

        self.initialize()
        self.task_completion = False
        self.current_steps = 0
        self.episode = self.episode + 1
        return self.state, self._get_info()

    def is_complete(self, user_response):
        user_concepts = self.get_user_concepts(user_response)
        if self.target in user_concepts:
            return True
        return False

    def context_encoder(self, context):
        context_flatten = ''
        for line in context:
            if line['role'] == 'system':
                continue
            elif line['role'] == 'user':
                context_flatten += "[user]{}".format(line['content'])
            elif line['role'] == 'assistant':
                context_flatten += "[chatbot]{}".format(line['content'])
        context_encoded = self.sentence_encoder.encode(context_flatten, convert_to_tensor=True)
        return context_encoded

    def _get_info(self):
        return {
            'episode': self.episode,
            'step': self.current_steps,
            'user response': self.previous_user_response,
            'reward': self.step_reward
        }

    def get_user_concepts(self, user_response):
        msg_concepts = concept_extractor(user_response)
        user_concepts = []
        for msg_concept in msg_concepts:
            if self.conceptnet.conceptnet.has_node(msg_concept):
                user_concepts.append(msg_concept)
        return user_concepts

    def render(self, mode='human'):
        pass
