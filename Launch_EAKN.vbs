Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\eaknnewproject\"
WshShell.Run "pythonw update.pyw", 0, False 
WshShell.Run "index.html"