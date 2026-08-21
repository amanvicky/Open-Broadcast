; OpenBroadcast - Eye Gaze Correction
; Inno Setup Script

#define MyAppName "OpenBroadcast"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Aman Vicky"
#define MyAppURL "https://github.com/amanvicky/Open-Broadcast"
#define MyAppExeName "main.py"

[Setup]
AppId={{B5E3F8A2-4C7D-4E9F-A1B2-C3D4E5F6A7B8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist
OutputBaseFilename=OpenBroadcast_Setup_{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Python embeddable package
Source: "..\dist\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

; Application files
Source: "..\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

; Core modules
Source: "..\core\*"; DestDir: "{app}\core"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\utils\*"; DestDir: "{app}\utils"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs

; Models directory (empty, created at runtime)
Source: "..\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Dirs]
Name: "{app}\models"
Name: "{app}\recordings"
Name: "{app}\data\raw"
Name: "{app}\data\pairs"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\run.bat"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\python\python.exe"; Parameters: "-m pip install -r requirements.txt --quiet"; StatusMsg: "Installing dependencies..."; Flags: runhidden waituntilterminated
Filename: "{app}\run.bat"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\models\__pycache__"
Type: filesandordirs; Name: "{app}\core\__pycache__"
Type: filesandordirs; Name: "{app}\ui\__pycache__"
Type: filesandordirs; Name: "{app}\utils\__pycache__"
Type: filesandordirs; Name: "{app}\data\__pycache__"
Type: filesandordirs; Name: "{app}\recordings"
