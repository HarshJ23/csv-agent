import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()  
ai_api_key = os.getenv('OPENAI_API_KEY')

# function to handle gsheet url input from user 
def csv_url_handler(sheet_url):
    sheet_id=sheet_url.split('/d/')[1].split('/')[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=sample.csv"
    return csv_url


sheet_url = input("Please enter the Google Sheets URL: ")
csv_path = csv_url_handler(sheet_url)


llm = ChatOpenAI(model="gpt-3.5-turbo" , temperature=0 , api_key=ai_api_key)

csv_agent = create_csv_agent(
 llm = llm ,
 path = csv_path,
 verbose=True,
return_intermediate_steps=True,
agent_type=AgentType.OPENAI_FUNCTIONS,
)


result = csv_agent.invoke("What is the gender distribution of the data in file?")

print("Result:")
print("Final output:" ,result['output'])

intermediate_steps = result['intermediate_steps']
query = intermediate_steps[0][0].tool_input['query']

print("Intermediate steps:" , intermediate_steps)
print("Query:", query)
