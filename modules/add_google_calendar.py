import datetime 
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def add_calandar(apply_date="2024-04-22"):
  """Shows basic usage of the Google Calendar API.
  Prints the start and name of the next 10 events on the user's calendar.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)

    ## Event add test

    ### Apply date
    date_var = datetime.datetime.strptime(apply_date, "%Y-%m-%d") 
    date_var = date_var + datetime.timedelta(days=0)
    date_var = date_var.strftime("%Y-%m-%d")            
    event = {
      'summary': 'Loan Apply date',
      'location': 'Seoul',
      'description': 'Loan ',
      'start': {
        'dateTime': '{}T09:00:00-07:00'.format(date_var),
        'timeZone': 'America/Los_Angeles',
      },
      'end': {
        'dateTime': '{}T17:00:00-07:00'.format(date_var),
        'timeZone': 'America/Los_Angeles',
      },
      'recurrence': []
      ,
      'attendees': [
      ],
      'reminders': {
        'useDefault': False,
        'overrides': [
          {'method': 'email', 'minutes': 24 * 60},
          {'method': 'popup', 'minutes': 10},
        ],
      },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()

    ### Analysis Date
      
    date_var = datetime.datetime.strptime(apply_date, "%Y-%m-%d") + datetime.timedelta(days=10)
    end_var = datetime.datetime.strptime(apply_date, "%Y-%m-%d") + datetime.timedelta(days=15)
    date_var = date_var.strftime("%Y-%m-%d")         
    end_var = end_var.strftime("%Y-%m-%d")  
      
    event = {
      'summary': 'Loan credit analysis date',
      'location': 'Seoul',
      'description': 'Loan ',
      'start': {
        'dateTime': '{}T09:00:00-07:00'.format(date_var),
        'timeZone': 'America/Los_Angeles',
      },
      'end': {
        'dateTime': '{}T17:00:00-07:00'.format(end_var),
        'timeZone': 'America/Los_Angeles',
      },
      'recurrence': []
      ,
      'attendees': [
      ],
      'reminders': {
        'useDefault': False,
        'overrides': [
          {'method': 'email', 'minutes': 24 * 60},
          {'method': 'popup', 'minutes': 10},
        ],
      },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()


    ### Execution date
    date_var = datetime.datetime.strptime(apply_date, "%Y-%m-%d") + datetime.timedelta(days=20)
    date_var = date_var.strftime("%Y-%m-%d")           
    event = {
      'summary': 'Loan execution date',
      'location': 'Seoul',
      'description': 'Loan ',
      'start': {
        'dateTime': '{}T09:00:00-07:00'.format(date_var),
        'timeZone': 'America/Los_Angeles',
      },
      'end': {
        'dateTime': '{}T17:00:00-07:00'.format(date_var),
        'timeZone': 'America/Los_Angeles',
      },
      'recurrence': []
      ,
      'attendees': [
      ],
      'reminders': {
        'useDefault': False,
        'overrides': [
          {'method': 'email', 'minutes': 24 * 60},
          {'method': 'popup', 'minutes': 10},
        ],
      },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()


    ### Execution Date
  except HttpError as error:
    print(f"An error occurred: {error}")


if __name__ == "__main__":
  add_calandar()

