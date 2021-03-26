import sqlite3
from sqlite3 import Error

class DB_SQLITE3():
    
    def __init__(self, db_path, check_same_thread=False):
        
        self.connection = self.__connect__(db_path, check_same_thread)
        
    def __connect__(self, db_path, check_same_thread):
        try:
            connection = sqlite3.connect(db_path, check_same_thread=check_same_thread)
            return connection
        except Error as e:
            print(e)
     
        return None
    
    def __del__(self):
        
        self.connection.close()
    
    def create_table(self, table_name, dict_type_fields, drop_exist=False):
        
        if self.connection == None:
            raise NotImplementedError('connection of database is not exists')
        
        cursor = self.connection.cursor()
        
        if drop_exist:
            query = "DROP TABLE IF EXISTS %s" % (table_name)
            cursor.execute(query)
        
        query = "CREATE TABLE %s (id INTEGER PRIMARY KEY AUTOINCREMENT, " % (table_name)
        
        for i, field in enumerate(dict_type_fields.keys()):
            query += field + ' ' + dict_type_fields[field]
            query += ', ' if i+1 != len(dict_type_fields.keys()) else '); '
            
        cursor.execute(query)
        self.connection.commit()
        
    def insert(self, table_name, dict_fields):
        
        query = "INSERT INTO %s (" % (table_name)
        prepare_statement = ''
        list_data = []
        
        fields = self.get_fields(table_name)
        
        for field in fields:
            query += field
            prepare_statement += '?'
            
            if field != fields[-1]:
                query += ', '
                prepare_statement += ', '
            else:
                query += ') '
                
            list_data.append(dict_fields[field] if field in dict_fields else None)
        
        query += 'VALUES (' + prepare_statement + ')'
        
        cursor = self.connection.cursor()
        cursor.execute(query, list_data)
        cursor.close()
        #self.connection.commit()
        
    
    def get_fields(self, table_name):
        
        cursor = self.connection.execute("SELECT * FROM %s" % (table_name))
        # instead of cursor.description:
        fields = [d[0] for d in cursor.description]
        cursor.close()
        
        return fields
    
    def select_by_id(self, table_name, id):
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM %s WHERE id=%s" % (table_name, id))
        row = cursor.fetchone()
        cursor.close()
    
        return row
    
    def select_by_cond(self, table_name, list_selected=[], dict_condition={}, list_condition_operation=None, list_operation=None):
        
        #list_selected = [field1, field2]
        #dict_condition = {'cond1':value1} #default = no condition
        #list_condition_operation = ['like', '='] #default is = (equal)
        #list_operation = [opt1, opt2, opt3] #default is AND
        
        fields = self.get_fields(table_name)
        query = "SELECT "
        
        for selected in list_selected:
            if selected in fields:
                query += selected
                if selected != list_selected[-1]:
                    query += ", "
                else:
                    query += " "
                
        
        if query == "SELECT ":
            query = "SELECT * "
        
        if not dict_condition:
            query += "FROM %s" % table_name
        else:
            query += "FROM %s WHERE " % table_name
        
        for i, cond in enumerate(dict_condition.keys()):
            query += cond + " " + (str(list_condition_operation[i]) if list_condition_operation != None else "=") + ' ? '
            
            if len(dict_condition.keys()) != i + 1:
                query += (list_operation[i] if list_operation != None else 'AND') + ' '
            
        cursor = self.connection.cursor()
        cursor.execute(query, list(dict_condition.values()))
        
        return cursor.fetchall()
        
    
    def update_by_id(self, table_name, id, dict_update):
        
        # dict_update = {'key1':value1, 'key2':value2}
        
        fields = self.get_fields(table_name)
        fields = fields[1:] # remove id (first field)
        update_data = {}
        
        for input_field in dict_update:
            
            if input_field in fields:
                update_data[input_field] = dict_update[input_field]
        
        query = "UPDATE %s SET " % (table_name)
        
        for i, update_field in enumerate(update_data):
            
            query += update_field + "=:" + update_field + (", " if i + 1 != len(update_data) else " ")
            
        query += "WHERE id=:id"
        
        update_data['id'] = id
        
        cursor = self.connection.cursor()
        #cursor.execute("update dataset_info set type=:type where id=1", {'type':'Cardiomegaly'})
        cursor.execute(query, update_data)
        #self.connection.commit()
                  
    def is_connect(self):
        
        return True if self.connection != None else False
    
    def count(self, table_name):
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM %s" % table_name)
        
        return cursor.fetchone()[0]
    
    def has_table(self, table_name):
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master where type='table' and name=?", (table_name,))
        exist_table = True if cursor.fetchone()[0] != None else False
        cursor.close()
        
        return exist_table
    
    def commit(self):
        
        self.connection.commit()
        
    
if __name__ == "__main__":
    
    db = DB_SQLITE3('./config/deep_xray.db')
    #print(db.is_connect())
    #print(db.select_by_id('dataset_info', 1))
    #print(db.get_fields('dataset_info'))
    #print(db.update_by_id('dataset_info', 1, {"using_type": "train"}))
    #print(db.count('dataset_info'))
    print(db.select_by_cond('dataset_info', ['id']))
