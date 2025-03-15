from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI

from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""





class PeerBusinessDeveloperAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.OpenAIGPT4oMini = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def investment_metrics_business_developer(self):
        return Agent(
            role="Investment Metrics Business Developer",
            backstory=dedent(
                f"""I have been selling Style Analytics solutions such as the Factor Skyline to asset owners in the US almost since the creation of the company in the 90s.
                I have been working extensivly with Robert Schwob to get the compùany started and find the first institutional clients.
                I stayed with the company when it was acquired by Investment Metrics. I have a deep understanding of how a pure analytics product can be useful to investors.
                """),
            goal=dedent(f"""
                        Create a compelling use case to present my portfolio analysis and reporting product to a potential client.
                        """),
            tools=[
                SearchTools.search_internet
            ],
            verbose=True,
            llm=self.OpenAIGPT4oMini,
        )

    def novus_business_developer(self):
        return Agent(
            role="Novus Business Developer",
            backstory=dedent(
                f"""I was hired at Novus Partners in 2010, before the purchase by SEI. I was the one who convinced SEI to acquire the company in order to boost analytical capabilities.
                I have been responsible to sell the Novus portfolio solutions to pure equity investors i.e. not multi asset investors.
                I have a deep understanding of how a pure analytics product can be useful to investors."""),
            goal=dedent(
                f"""Create a compelling use case to present my advanced portfolio management dashboards and investment operations tools to a potential client."""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.OpenAIGPT4oMini,
        )

class PotentialClientAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.OpenAIGPT4oMini = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def ao_decision_maker(self):
        return Agent(
            role="C-Suite Decision Maker at Major Pension Fund",
            backstory=dedent(
                f"""I am running the equity department at a large Pension Fund in Europe, I have 30 years of experience in the industry.
                I had several roles before joining my Pension Fund, most recently as an equities portfolio manager with a large Asset Manager.
                I have had experience selecting smart beta managers and indices and more recently ESG solutions with low tracking error and few risk deviations.
                At my Pension Fund, I have been responsible for deploying a new ESG do-no-harm policy leading to the exclusion of several companies from the portfolio.
                """),
            goal=dedent(f"""
                        Provide an honest feedback on opportunities to purchase a portfolio analysis and reporting product.
                        """),
            tools=[
                SearchTools.search_internet
            ],
            verbose=True,
            llm=self.OpenAIGPT4oMini,
        )

    def am_business_developer(self):
        return Agent(
            role="Amundi Business Developer",
            backstory=dedent(
                f"""I am Fannie Wurtz, I have been running Amundi's ETF business since 2013.
                We hace developed a technology platform called ALTO which is not that great to support the business development effort, it is actually good for execution of trades.
                """),
            goal=dedent(
                f"""Look into portfolio analysis and reporting products and assess their interest for the promotion of Amundi's equity funds and ETFs."""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.OpenAIGPT4oMini,
        )
    
class ScientificPortfolioAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.OpenAIGPT4oMini = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def ceo(self):
        return Agent(
            role="CEO of Scientific Portfolio",
            backstory=dedent(
                f"""I am Benjamin Herzog, CEO of Scientific Portfolio. I have been in the EDHEC ecosystem for 5 years and before that I spent 15 years at Societe Generale.
                I have an engineering background and I have been working on the development of the Scientific Beta indices and the EDHEC Risk Institute.
                I have worked on the development of this platform under the hood for 3Y and can't wait to find a first booming business case.
                """),
            goal=dedent(f"""
                        Our portfolio analysis platform is great but remains a nice-to-have.
                        We urgently need to find use cases making it a must-have.
                        In order to do that, I would like to discuss with business developers from comparable companies and discuss the use cases they see as effective
                        from their discussions and successfuly pitches to clients in our target.
                        """),
            tools=[
                SearchTools.search_internet
            ],
            verbose=True,
            llm=self.OpenAIGPT4oMini,
        )
