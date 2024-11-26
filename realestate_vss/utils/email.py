from typing import List
import ssl
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

def send_email_alert(subject: str, html_content: str, sender_email: str, receiver_emails: List[str], password: str):
  """
  Send an HTML email alert using Gmail SMTP.
  """
  msg = MIMEMultipart()
  msg['Subject'] = subject
  msg['From'] = sender_email
  msg['To'] = ", ".join(receiver_emails)
  
  body_html = MIMEText(html_content, 'html')
  msg.attach(body_html)

  smtp_server = "smtp.gmail.com"
  port = 587
  context = ssl.create_default_context()
  
  try:
    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_emails, msg.as_string())
    logger.info('Alert email sent successfully')
  except Exception as e:
    logger.error(f"Error sending email: {e}")
  finally:
    server.quit()