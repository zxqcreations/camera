import usb.core
import usb.util

dev = usb.core.find(find_all = True)

print(dev)

if dev is None:
    print('failed to connect')
    
for cfg in dev:
    print('123')
    print('prd' + str(cfg.idProduct))
    print(str(usb.util.get_string(cfg, 256, cfg.iManufacturer)))


